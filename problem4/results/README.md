1.Batch_size = 128 <br><br>
2.每隔一段时间计算 training set 和 validation set 上的 precision, recall, f1_score, 分别保存在train_p_r_f1.txt 和 val_p_r_f1.txt中, 
找到最佳的parameters, 保存在model_params.pkl中（github有25M的上传限制，model_parameters太大，上传不了）<br><br>
3.val_p_r_f1.txt中第4列保存的是当前找到的最优f1_score<br><br>
4.test_p_r_f1.txt保存的是最后的测试结果<br><br>
5.其余results保存的是每次计算后所得的序列标注结果(分为iob1和iobes两种模式，去掉了空行)
