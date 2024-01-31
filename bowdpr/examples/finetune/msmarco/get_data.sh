wget --no-check-certificate https://rocketqa.bj.bcebos.com/corpus/marco.tar.gz
tar -zxf marco.tar.gz
cd marco

wget https://msmarco.blob.core.windows.net/msmarcoranking/qidpidtriples.train.full.2.tsv.gz
wget https://msmarco.blob.core.windows.net/msmarcoranking/qrels.train.tsv -O qrels.train.tsv
gunzip qidpidtriples.train.full.2.tsv.gz
join  -t "$(echo -en '\t')"  -e '' -a 1  -o 1.1 2.2 1.2  <(sort -k1,1 para.txt) <(sort -k1,1 para.title.txt) | sort -k1,1 -n > corpus.tsv
awk -v RS='\r\n' '$1==last {printf ",%s",$3; next} NR>1 {print "";} {last=$1; printf "%s\t%s",$1,$3;} END{print "";}' qidpidtriples.train.full.2.tsv > train.negatives.tsv

# Text format
python build_train_jsonl.py --negative_file train.negatives.tsv --qrels qrels.train.tsv --n_sample 128 --queries train.query.txt --collection corpus.tsv --save_to text/train
python build_query_passage_text.py --file dev.query.txt --save_to text/query/dev.query.jsonl --is_query
python build_query_passage_text.py --file train.query.txt --save_to text/query/train.query.jsonl --is_query
python build_query_passage_text.py --n_splits 8 --file corpus.tsv --save_to text/corpus

cd ..
