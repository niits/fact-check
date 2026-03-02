# mkdir -p datas
# mkdir -p models
# wget -O datas/train.jsonl https://fever.ai/download/feverous/feverous_train_challenges.jsonl
# wget -O datas/dev.jsonl https://fever.ai/download/feverous/feverous_dev_challenges.jsonl
# wget -O datas/test_unlabeled.jsonl https://fever.ai/download/feverous/feverous_test_unlabeled.jsonl
wget -O datas/feverous-wiki-pages-db.zip https://fever.ai/download/feverous/feverous-wiki-pages-db.zip
unzip datas/feverous-wiki-pages-db.zip -j datas/feverous_wikiv1.db