# Img-classifier

保存一些用于最基础图片分类的模型，训练验证以及预测代码。包括

- Lr
- vgg
- Lenet5
- Alexnet
- resnet
- Vit

以及vit后续各种sota模型训练方法。

Requirement:
======

	Python: 3.7   
	numpy
	pandas
	pytorch
	transformers

data:
======

数据暂时使用 mnist手写数字识别，cifar10等常用数据集。



How to run the code?
====

#### train：

1. 将训练集，验证集以及ner的label文件存入 data中。

2. 将预训练模型保存入，pretrain_model中。

3. 修改code/script中对应框架下的shell文件

   如 run_ner_gp.sh

   ```sh
   CURRENT_DIR=`pwd`
   export BERT_BASE_DIR=../pretrain_model/roberta-wwm-chinese
   export OUTPUR_DIR=../outputs/gp
   TASK_NAME="queryner"
   #
   python run_ner_gp.py \
     --model_type=bert \
     --train_data_path ../data/test_data/train.csv \
     --dev_data_path ../data/test_data/dev.csv \
     --label_txt ../data/label_dir/label.txt \
     --model_name_or_path=$BERT_BASE_DIR \
     --task_name=$TASK_NAME \
     --do_eval \
     --do_lower_case \
     --train_max_seq_length=128 \
     --eval_max_seq_length=512 \
     --per_gpu_train_batch_size=24 \
     --per_gpu_eval_batch_size=24 \
     --learning_rate=3e-5 \
     --crf_learning_rate=1e-3 \
     --num_train_epochs=4.0 \
     --logging_steps=-1 \
     --save_steps=-1 \
     --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
     --overwrite_output_dir \
     --seed=42
   ```

4. 运行训练脚本

   ```sh
   sh script/run_ner_gp.sh
   ```

5. 参数设置

   - model_name_or_path 预训练模型地址
   - do_adv 使用fgm进行对抗训练
   - label_txt 标签txt文件地址
   - train_data_path 训练数据地址
   - dev_data_path 验证集地址

#### Predict:

1. 运行sh脚本，如：predict_ner_gp.sh

   ```shell
   export BERT_BASE_DIR=../outputs/gp/queryner_output_0515_full/bert/checkpoint-38685
   export OUTPUR_DIR=../outputs/gp
   TASK_NAME="queryner"
   #
   python run_ner_gp.py \
     --model_type=bert \
     --predict_data_path ../data/red_spu_left.csv  \
     --result_data_path ../data/0523_span.csv \
     --save_type span_csv \
     --label_txt ../data/label_dir/label_p0.txt \
     --model_name_or_path=$BERT_BASE_DIR \
     --task_name=$TASK_NAME \
     --do_predict \
     --do_lower_case \
     --eval_max_seq_length 32 \
     --per_gpu_eval_batch_size 256 \
     --save_steps=-1 \
     --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
     --overwrite_output_dir \
     --seed=42
   
   ```

2. 参数设置

   - predict_data_path 待预测文件，可以为csv文件（query 列为待预测text）或txt文件
   - save_type可选（span_csv,span_json,bio_csv,bio_txt) 4种类型作为结果保存文件。
   - eval_max_seq_length 分词的max_length参数，根据输入文本确定，影响预测速度。



## Acknowledge: 

参考网上开源项目：

- 