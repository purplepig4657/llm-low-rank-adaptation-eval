import jsonlines
import torch
import gc

from eval.math.util import util
from vllm import LLM, SamplingParams
import sys
MAX_INT = sys.maxsize
INVALID_ANS = "[invalid]"

class MATHEval:
    def __init__(
        self, 
        model,  # model path
        data_path = "eval/math/MATH/data/MATH_test.jsonl", 
        start=0, 
        end=MAX_INT, 
        batch_size=50, 
        tensor_parallel_size=1
    ):
        self.model = model
        self.data_path = data_path
        self.start = start
        self.end = end
        self.batch_size = batch_size
        self.tensor_parallel_size = tensor_parallel_size
        self.invalid_outputs = []

    def remove_boxed(self, s):
        left = "\\boxed{"
        try:
            assert s[:len(left)] == left
            assert s[-1] == "}"
            return s[len(left):-1]
        except:
            return None

    def process_results(self, doc, completion, answer):
        split_ans = completion.split('The answer is: ')
        if len(split_ans) > 1:
            ans = split_ans[-1]
            extract_ans_temp = ans.split('.\n')[0]
            extract_ans_temp = extract_ans_temp.strip()
            if len(extract_ans_temp)>0 and extract_ans_temp[-1] == '.':
                extract_ans = extract_ans_temp[0:-1]
            else:
                extract_ans = extract_ans_temp
            extract_ans = extract_ans.strip()
            if util.is_equiv(extract_ans, answer):
                return True
            else:
                return False
        else:
            temp = {'question': doc, 'output': completion, 'answer': answer}
            self.invalid_outputs.append(temp)
            return False

    def batch_data(self, data_list, batch_size=1):
        n = len(data_list) // batch_size
        batch_data = []
        for i in range(n-1):
            start = i * batch_size
            end = (i+1)*batch_size
            batch_data.append(data_list[start:end])

        last_start = (n-1) * batch_size
        last_end = MAX_INT
        batch_data.append(data_list[last_start:last_end])
        return batch_data

    def test_hendrycks_math(self):
        hendrycks_math_ins = []
        hendrycks_math_answers = []
        problem_prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
        )
        print('promt =====', problem_prompt)
        with open(self.data_path, "r+", encoding="utf8") as f:
            for idx, item in enumerate(jsonlines.Reader(f)):
                temp_instr = problem_prompt.format(instruction=item["instruction"])
                hendrycks_math_ins.append(temp_instr)
                solution = item['output']
                temp_ans = self.remove_boxed(util.last_boxed_only_string(solution))
                hendrycks_math_answers.append(temp_ans)

        print('total length ===', len(hendrycks_math_ins))
        hendrycks_math_ins = hendrycks_math_ins[self.start:self.end]
        hendrycks_math_answers = hendrycks_math_answers[self.start:self.end]
        print('lenght ====', len(hendrycks_math_ins))
        batch_hendrycks_math_ins = self.batch_data(hendrycks_math_ins, batch_size=self.batch_size)

        stop_tokens = ["Instruction:", "Instruction", "Response:", "Response"]
        sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=2048, stop=stop_tokens)
        print('sampleing =====', sampling_params)
        llm = LLM(model=self.model,tensor_parallel_size=self.tensor_parallel_size,gpu_memory_utilization=0.8,trust_remote_code=True)
        res_completions = []
        for idx, (prompt, prompt_answer) in enumerate(zip(batch_hendrycks_math_ins, hendrycks_math_answers)):
            if isinstance(prompt, list):
                pass
            else:
                prompt = [prompt]
            completions = llm.generate(prompt, sampling_params)
            for output in completions:
                prompt_temp = output.prompt
                generated_text = output.outputs[0].text
                res_completions.append(generated_text)

        results = []
        for idx, (prompt, completion, prompt_answer) in enumerate(zip(hendrycks_math_ins, res_completions, hendrycks_math_answers)):
            res = self.process_results(prompt, completion, prompt_answer)
            results.append(res)

        acc = sum(results) / len(results)
        print('valid_outputs===', self.invalid_outputs)
        print('len invalid outputs ====', len(self.invalid_outputs))
        print('start===', self.start, ', end====',self.end)
        print('length====', len(results), ', acc====', acc)
        return {'metric': 'acc', 'value': acc}

    def eval(self):
        return self.test_hendrycks_math()

    def cleanup(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    del obj
            except:
                pass
            
        gc.collect()
