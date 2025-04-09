import argparse
import json
import re
import jsonlines
from fraction import Fraction
from vllm import LLM, SamplingParams
import sys
from eval.math.util.grader import math_equal
MAX_INT = sys.maxsize
import transformers
import torch
import gc

class GSM8KEval:
    def __init__(
        self, 
        model,  # model path
        data_path = "eval/math/GSM8K/data/gsm8k_test.jsonl", 
        start=0, 
        end=MAX_INT, 
        batch_size=60, 
        tensor_parallel_size=1,
    ):
        self.model = model
        self.data_path = data_path
        self.start = start
        self.end = end
        self.batch_size = batch_size
        self.tensor_parallel_size = tensor_parallel_size

    def is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            pass
        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass
        return False

    def extract_answer_number(self, completion):
        text = completion.split('The answer is: ')
        if len(text) > 1:
            extract_ans = text[-1].strip()
            match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
            if match:
                if '/' in match.group():
                    denominator = match.group().split('/')[1]
                    numerator = match.group().split('/')[0]
                    if self.is_number(denominator) == True and self.is_number(numerator) == True:
                        if denominator == '0':
                            return round(float(numerator.replace(',', '')))
                        else:
                            frac = Fraction(match.group().replace(',', ''))
                            num_numerator = frac.numerator
                            num_denominator = frac.denominator
                            return round(float(num_numerator / num_denominator))
                    else:
                        return None
                else:
                    if float(match.group().replace(',', '')) == float('inf'):
                        return None
                    return round(float(match.group().replace(',', '')))
            else:
                return None
        else:
            return None

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


    def gsm8k_test(self):
        INVALID_ANS = "[invalid]"
        gsm8k_ins = []
        gsm8k_answers = []
        problem_prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
        )
        print('promt =====', problem_prompt)
        with open(self.data_path,"r+", encoding="utf8") as f:
            for idx, item in enumerate(jsonlines.Reader(f)):
                temp_instr = problem_prompt.format(instruction=item["question"])
                gsm8k_ins.append(temp_instr)
                temp_ans = item['answer'].split('#### ')[1]
                temp_ans = int(temp_ans.replace(',', ''))
                gsm8k_answers.append(temp_ans)

        gsm8k_ins = gsm8k_ins[self.start:self.end]
        gsm8k_answers = gsm8k_answers[self.start:self.end]
        print('lenght ====', len(gsm8k_ins))
        batch_gsm8k_ins = self.batch_data(gsm8k_ins, batch_size=self.batch_size)

        stop_tokens = ["Instruction:", "Instruction", "Response:", "Response"]
        sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=1024, stop=stop_tokens)
        print('sampleing =====', sampling_params)
        llm = LLM(model=self.model,tensor_parallel_size=self.tensor_parallel_size,gpu_memory_utilization=0.8,trust_remote_code=True) #
        #llm = transformers.AutoModelForCausalLM.from_pretrained(
        #    model,
        #    #torch_dtype=torch.bfloat16,
        #    device_map="auto",
        #    trust_remote_code=True
        #)    
        result = []
        res_completions = []
        for idx, (prompt, prompt_answer) in enumerate(zip(batch_gsm8k_ins, gsm8k_answers)):
            if isinstance(prompt, list):
                pass
            else:
                prompt = [prompt]

            completions = llm.generate(prompt, sampling_params)
            for output in completions:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                res_completions.append(generated_text)

        invalid_outputs = []
        for idx, (prompt, completion, prompt_answer) in enumerate(zip(gsm8k_ins, res_completions, gsm8k_answers)):
            doc = {'question': prompt}
            y_pred = self.extract_answer_number(completion)
            if y_pred != None:
                result.append(float(y_pred) == float(prompt_answer) or math_equal(y_pred, prompt_answer))
            else:
                result.append(False)
                temp = {'question': prompt, 'output': completion, 'answer': prompt_answer}
                invalid_outputs.append(temp)
        acc = sum(result) / len(result)
        print('len invalid outputs ====', len(invalid_outputs), ', valid_outputs===', invalid_outputs)
        print('start===', self.start, ', end====', self.end)
        print('gsm8k length====', len(result), ', gsm8k acc====', acc)
        return acc


    def eval(self):
        acc = self.gsm8k_test()
        return {'metric': 'acc', 'acc': acc}

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
