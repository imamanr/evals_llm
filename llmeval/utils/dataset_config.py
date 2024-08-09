from ..smp.misc import listinstr

dataset_URLs = {
    
    # 'DocVQA_VAL': 'https://opencompass.openxlab.space/utils/VLMEval/DocVQA_VAL.tsv',
    # 'DocVQA_TEST': 'https://opencompass.openxlab.space/utils/VLMEval/DocVQA_TEST.tsv',
    # 'InfoVQA_VAL': 'https://opencompass.openxlab.space/utils/VLMEval/InfoVQA_VAL.tsv',
    # 'InfoVQA_TEST': 'https://opencompass.openxlab.space/utils/VLMEval/InfoVQA_TEST.tsv',
    # 'AI2D_TEST': 'https://opencompass.openxlab.space/utils/VLMEval/AI2D_TEST.tsv',
    # 'LLaVABench': 'https://opencompass.openxlab.space/utils/VLMEval/LLaVABench.tsv',
    # 'OCRBench': 'https://opencompass.openxlab.space/utils/VLMEval/OCRBench.tsv',
    # 'ChartQA_TEST': 'https://opencompass.openxlab.space/utils/VLMEval/ChartQA_TEST.tsv',
    # 'MMStar': 'https://opencompass.openxlab.space/utils/VLMEval/MMStar.tsv',
    # 'RealWorldQA': 'https://opencompass.openxlab.space/utils/VLMEval/RealWorldQA.tsv',
    # 'POPE': 'https://opencompass.openxlab.space/utils/VLMEval/POPE.tsv',
}

dataset_md5_dict = {
   # 'POPE': 'c12f5acb142f2ef1f85a26ba2fbe41d5',
}

img_root_map = {k: k for k in dataset_URLs}
img_root_map.update({
   
    # 'COCO_VAL': 'COCO',
    # 'OCRVQA_TEST': 'OCRVQA',
    # 'OCRVQA_TESTCORE': 'OCRVQA',
    # 'TextVQA_VAL': 'TextVQA',
 
})

#assert set(dataset_URLs) == set(img_root_map)


def DATASET_TYPE(dataset):
    # Dealing with Custom Dataset
    dataset = dataset.lower()
    if listinstr(['mmbench', 'seedbench', 'ccbench', 'mmmu', 'scienceqa', 'ai2d', 'mmstar', 'realworldqa'], dataset):
        return 'multi-choice'
    elif listinstr(['mme', 'hallusion', 'pope'], dataset):
        return 'Y/N'
    elif 'coco' in dataset:
        return 'Caption'
    elif listinstr(['ocrvqa', 'textvqa', 'chartqa', 'mathvista', 'docvqa', 'infovqa', 'llavabench',
                    'mmvet', 'ocrbench'], dataset):
        return 'VQA'
    else:
        if dataset not in dataset_URLs:
            import warnings
            warnings.warn(f"Dataset {dataset} not found in dataset_URLs, will use 'multi-choice' as the default TYPE.")
            return 'multi-choice'
        else:
            return 'QA'


def abbr2full(s):
    datasets = [x for x in img_root_map]
    ins = [s in d for d in datasets]
    if sum(ins) == 1:
        for d in datasets:
            if s in d:
                return d
    else:
        return s
