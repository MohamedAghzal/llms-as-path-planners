from helpers import fixed_length, single_environment, group_environments, decompose_sample
from prompting import n_shot_prompt, next_example, VALS_OOD, VALS_IID
from openai import AzureOpenAI
import json
import time
from evaluate import success_sg
import sys

client = AzureOpenAI(
    azure_endpoint="",
    api_version="",
    api_key=""
)

def inference(prompt, model='gpt-4-turbo', message_texts = None):
    print(prompt)
    if(message_texts == None):
        message_text = [
            {
                "role":"system",
                "content":prompt
            }
        ]
    else:
        message_text = message_texts
    
    completion = client.chat.completions.create(
        model=model,  
        messages = message_text,  
        temperature=0.0,  
        max_tokens=200,  
        top_p=0.95,  
        frequency_penalty=0.25,  
        presence_penalty=0,  
        stop=None
    )

    return completion


def few_shot_inference(train, test_set, n_exemplars, representation):
    type_ = None
    if('AE' in representation):
         type_ = representation.split('_')[1]
         representation = 'AE'

    n_shot = n_shot_prompt(train, n_exemplars, None, AE_type=type_)

    out = []
    for ex in test_set:
        prompt = next_example(ex, n_shot=n_shot, representation=representation) 
        prompt = 'Your output throughout this conversation should only consists of tokens (left, right, up, down). ' + prompt
        try:
            response = inference(prompt)
        except:
            time.sleep(30)
            response = inference(prompt)
            pass
        
        predicted = response.choices[0].message.content
        print('FULL:', predicted)

        if('Thought' in predicted):
            try:
                thought, sequence = predicted.split('Solution: ')
            except:
                sequence = ''

        if('AE' not in representation):
            print(success_sg(ex['world'], predicted))

        out.append({
            'Grid': ex['grid_representation'],
            'Predicted': response.choices[0].message.content,
            'Correct': ex['path'],
            'World': ex['world']
        })
    return out


def main():
    '''
    CLA: 
    type of test
        - length: few-shot examples are of the same length as test samples
        - env: few-shot examples and test set are drawn from the same environment
    geometry:
        - rectangle
        - maze
        - zig_zag
    representation:
        - Naive
        - Code
        - Grid
        - AE
        - Grid2Grid
    '''

    choice = sys.argv[1]

    print(choice)
    geometry = sys.argv[2]
    representation = sys.argv[3]

    if(choice == 'env_from_file'):

        res_iid = []
        res_ood = []
        iid_file = open(sys.argv[4])
        ood_file = open(sys.argv[5])

        iid_data = json.load(iid_file)
        ood_data = json.load(ood_file)

        grouped =  group_environments(data=iid_data + ood_data, geometry=geometry)            

        n_environments = 30
        valid = {}
        for id_ in grouped.keys():
           print(len(grouped[id_]['OOD']))         
           if(len(grouped[id_]['OOD']) < 4):
               continue
           valid[id_] = grouped[id_]
        
        for id_ in list(valid.keys())[:n_environments]:
           print(f'Processing environment {id_}')
           test_samples = 5

           iid_values = VALS_IID[geometry]
           ood_values = VALS_OOD[geometry]

           count = {}
           
           test_iid = []
           for x in grouped[id_]['IID']:
               vv = len(x['path'].split())
               if(vv in count.keys()):
                   continue
               count[vv] = True
               test_iid.append(x)

           train = []

           for x in grouped[id_]['IID']:
               if(x not in test_iid):
                   train.append(x)
           
           test_ood = grouped[id_]['OOD'] 

           if(representation == 'Grid'):
                res_iid.append(few_shot_inference(train, test_iid, 5, 'Grid'))
                res_ood.append(few_shot_inference(train, test_ood, 5, 'Grid'))

           if(representation == 'Code'):
                res_iid.append(few_shot_inference(train, test_iid, 5, 'Code'))
                res_ood.append(few_shot_inference(train, test_ood, 5, 'Code')) 

           if('AE' in representation):
                res_iid.append(few_shot_inference(train, test_iid, 5, representation))
                res_ood.append(few_shot_inference(train, test_ood, 5, representation)) 
           
           if(representation == 'Naive'):
                res_iid.append(few_shot_inference(train, test_iid, 5, 'Naive'))
                res_ood.append(few_shot_inference(train, test_ood, 5, 'Naive'))  

           with open(f'outputs_fullSet/{choice}_out_5_shot_{geometry}_{representation}_iid_fewShot_15x15.json', 'w') as f:
                obj = json.dumps(res_iid, indent=4)
                f.write(obj)
           with open(f'outputs_fullSet/{choice}_out_5_shot_{geometry}_{representation}_ood_fewShot_15x15.json', 'w') as f:
                obj = json.dumps(res_ood, indent=4)
                f.write(obj)

    if(choice == 'decompose'):
            res_iid = []
            res_ood = []
            iid_file = open(sys.argv[4])
            ood_file = open(sys.argv[5])
            max_size = int(sys.argv[6])

            iid_data = json.load(iid_file)
            ood_data = json.load(ood_file)

            grouped =  group_environments(data=iid_data + ood_data, geometry=geometry)            

            n_environments = 30
            valid = {}
            for id_ in grouped.keys():
                if(len(grouped[id_]['OOD']) < 5):
                    continue
                valid[id_] = grouped[id_]
            
            for id_ in list(valid.keys())[:n_environments]:
                print(f'Processing environment {id_}')
                test_samples = 5

                iid_values = VALS_IID[geometry]
                ood_values = VALS_OOD[geometry]

                count = {}
                
                test_iid = []
                for x in grouped[id_]['IID']:
                    vv = len(x['path'].split())
                    if(vv in count.keys()):
                        continue
                    count[vv] = True
                    test_iid.append(x)

                train = []

                for x in grouped[id_]['IID']:
                    if(x not in test_iid):
                        train.append(x)
                
                test_ood = grouped[id_]['OOD'] 

                decomposed_iid = []
                decomposed_ood = []
                if(representation == 'Grid'):
                        for instance in test_iid:
                            compositions, _ = decompose_sample(instance, max_len=max_size, geometry=geometry)
                            decomposed_iid.append(few_shot_inference(train, compositions, 5, 'Grid'))
                        for instance in test_ood:
                            compositions, _ = decompose_sample(instance, max_len=max_size, geometry=geometry)
                            decomposed_ood.append(few_shot_inference(train, compositions, 5, 'Grid'))
                        res_iid.append(decomposed_iid)
                        res_ood.append(decomposed_ood)
                if(representation == 'Code'):
                        for instance in test_iid:
                            compositions, _ = decompose_sample(instance, max_len=max_size, geometry=geometry)
                            decomposed_iid.append(few_shot_inference(train, compositions, 5, 'Code'))
                        for instance in test_ood:
                            compositions, _ = decompose_sample(instance, max_len=max_size, geometry=geometry)
                            decomposed_ood.append(few_shot_inference(train, compositions, 5, 'Code'))

                        res_iid.append(decomposed_iid)
                        res_ood.append(decomposed_ood)

                if(representation == 'AE'):
                        for instance in test_iid:
                            compositions, _ = decompose_sample(instance, max_len=max_size, geometry=geometry)
                            decomposed_iid.append(few_shot_inference(train, compositions, 5, 'AE'))
                        for instance in test_ood:
                            compositions, _ = decompose_sample(instance, max_len=max_size, geometry=geometry)
                            decomposed_ood.append(few_shot_inference(train, compositions, 5, 'AE'))
                            
                        res_iid.append(decomposed_iid)
                        res_ood.append(decomposed_ood)

                if(representation == 'Naive'):
                        for instance in test_iid:
                            compositions, _ = decompose_sample(instance, max_len=max_size, geometry=geometry)
                            decomposed_iid.append(few_shot_inference(train, compositions, 5,'Naive'))
                        for instance in test_ood:
                            compositions, _ = decompose_sample(instance, max_len=max_size, geometry=geometry)
                            decomposed_ood.append(few_shot_inference(train, compositions, 5,'Naive'))
                            
                        res_iid.append(decomposed_iid)
                        res_ood.append(decomposed_ood)


                with open(f'../outputs/{choice}_out_5_shot_{geometry}_{representation}_iid_decomposed.json', 'w') as f:
                        obj = json.dumps(res_iid, indent=4)
                        f.write(obj)
                with open(f'../outputs/{choice}_out_5_shot_{geometry}_{representation}_ood_decomposed.json', 'w') as f:
                        obj = json.dumps(res_ood, indent=4)
                        f.write(obj)

if __name__ == '__main__':
    main()