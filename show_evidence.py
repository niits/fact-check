from src.modules.datasets.feverous import FeverousEvidenceFormat
import json

dataset = FeverousEvidenceFormat.from_path("/raid/Workspace/an/code/factcheck/FEVEROUS/data/dev.jsonl",
                             "/raid/Workspace/an/code/factcheck/FEVEROUS/data/feverous_wikiv1.db")

n = 0
for sample in dataset:
    print('--------------------------------')
    print('claim : ', sample.claim)
    print('evidence : \n')
    if sample.evidence:
        for ev in sample.evidence:
            print(ev.content)
            if ev.context:
                print('  context:', ev.context)
    print('label : ', sample.label)
    print('--------------------------------\n\n')
    n += 1
    if n > 100:
        break
