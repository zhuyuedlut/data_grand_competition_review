### self-training-bert-ext ce + NTLoss 0.5
nohup python src/bert_models/training/main.py --model_type bert --model_name_or_path resources/bert/bert-self-training-ext --data_dir ./datasets/phase_1/splits/fold_0 --label_file_level_1 datasets/phase_1/labels_level_1.txt --label_file_level_2 datasets/phase_1/labels_level_2.txt --task daguan --aggregator bert_pooler --model_dir ./experiments/outputs/daguan/self_bert_0917_1 --do_train --do_eval --train_batch_size 32 --num_train_epochs 50 --embeddings_learning_rate 1e-5 --encoder_learning_rate 1e-5 --classifier_learning_rate 1e-4 --warmup_steps 200 --max_seq_len 132 --dropout_rate 0.15 --metric_key_for_early_stop "macro avg__f1-score__level_2" --logging_steps 200 --patience 6 --label2freq_level_1_dir datasets/phase_1/label2freq_level_1.json --label2freq_level_2_dir datasets/phase_1/label2freq_level_2.json --loss_fct_name ce --use_class_weights --contrastive_loss ntxent_loss --what_to_contrast sample_and_class_embeddings --use_ms_dropout > ./experiments/logs/bert_0917_1.log &

### self-training-bert-ext ce + NTLoss 0.5 + 对比学习
nohup python src/bert_models/training/main.py --model_type bert --model_name_or_path resources/bert/bert-self-training-ext --data_dir ./datasets/phase_1/splits/fold_0 --label_file_level_1 datasets/phase_1/labels_level_1.txt --label_file_level_2 datasets/phase_1/labels_level_2.txt --task daguan --aggregator bert_pooler --model_dir ./experiments/outputs/daguan/self_bert_0918_1 --do_train --do_eval --train_batch_size 32 --num_train_epochs 50 --embeddings_learning_rate 1e-5 --encoder_learning_rate 1e-5 --classifier_learning_rate 1e-4 --warmup_steps 200 --max_seq_len 132 --dropout_rate 0.15 --metric_key_for_early_stop "macro avg__f1-score__level_2" --logging_steps 200 --patience 6 --label2freq_level_1_dir datasets/phase_1/label2freq_level_1.json --label2freq_level_2_dir datasets/phase_1/label2freq_level_2.json --loss_fct_name ce --use_class_weights --contrastive_loss ntxent_loss --what_to_contrast sample_and_class_embeddings --use_ms_dropout --at_method pgd > ./experiments/logs/bert_0918_1.log &
### self-training-bert-ext dice + NTLoss 0.5 
nohup python src/bert_models/training/main.py --model_type bert --model_name_or_path resources/bert/bert-self-training-ext --data_dir ./datasets/phase_1/splits/fold_0 --label_file_level_1 datasets/phase_1/labels_level_1.txt --label_file_level_2 datasets/phase_1/labels_level_2.txt --task daguan --aggregator bert_pooler --model_dir ./experiments/outputs/daguan/self_bert_0921_1 --do_train --do_eval --train_batch_size 32 --num_train_epochs 50 --embeddings_learning_rate 0.5e-4 --encoder_learning_rate 0.5e-4 --classifier_learning_rate 5e-4 --warmup_steps 200 --max_seq_len 132 --dropout_rate 0.25 --metric_key_for_early_stop "macro avg__f1-score__level_2" --logging_steps 200 --patience 6 --label2freq_level_1_dir datasets/phase_1/label2freq_level_1.json --label2freq_level_2_dir datasets/phase_1/label2freq_level_2.json --loss_fct_name focal --focal_loss_gamma 0.3 --contrastive_loss ntxent_loss --what_to_contrast sample_and_class_embeddings --use_ms_dropout > ./experiments/logs/bert_0921_1.log &

### self-training-bert-ext SimCLS
nohup python src/bert_models/training/main.py --model_type bert --model_name_or_path resources/bert/bert-self-training-ext --data_dir ./datasets/phase_1/splits/fold_0 --label_file_level_1 datasets/phase_1/labels_level_1.txt --label_file_level_2 datasets/phase_1/labels_level_2.txt --task daguan --aggregator bert_pooler --model_dir ./experiments/outputs/daguan/self_bert_0919_1 --do_train --do_eval --train_batch_size 32 --num_train_epochs 50 --embeddings_learning_rate 1e-5 --encoder_learning_rate 1e-5 --classifier_learning_rate 1e-4 --warmup_steps 200 --max_seq_len 132 --dropout_rate 0.15 --metric_key_for_early_stop "macro avg__f1-score__level_2" --logging_steps 200 --patience 6 --label2freq_level_1_dir datasets/phase_1/label2freq_level_1.json --label2freq_level_2_dir datasets/phase_1/label2freq_level_2.json --loss_fct_name ce --use_class_weights --contrastive_loss supconloss --what_to_contrast sample_and_class_embeddings --use_ms_dropout --ms_average > ./experiments/logs/bert_0919_1.log &

### self-training-bert-ext bertPee
nohup python src/bert_models/training/main.py --model_type bert_pabee --model_name_or_path resources/bert/bert-self-training-ext --data_dir ./datasets/phase_1/splits/fold_0 --label_file_level_1 datasets/phase_1/labels_level_1.txt --label_file_level_2 datasets/phase_1/labels_level_2.txt --task daguan --aggregator bert_pooler --model_dir ./experiments/outputs/daguan/self_bert_0919_2 --do_train --do_eval --train_batch_size 32 --num_train_epochs 50 --embeddings_learning_rate 1e-5 --encoder_learning_rate 1e-5 --classifier_learning_rate 1e-4 --warmup_steps 200 --max_seq_len 132 --dropout_rate 0.15 --metric_key_for_early_stop "macro avg__f1-score__level_2" --logging_steps 200 --patience 6 --label2freq_level_1_dir datasets/phase_1/label2freq_level_1.json --label2freq_level_2_dir datasets/phase_1/label2freq_level_2.json --loss_fct_name ce --use_class_weights --contrastive_loss supconloss --what_to_contrast sample_and_class_embeddings --use_ms_dropout > ./experiments/logs/bert_0919_2.log &