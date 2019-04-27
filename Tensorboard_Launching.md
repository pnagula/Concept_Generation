# First create reverse tunnel enabled ssh connection 
	gcloud compute ssh pnagula@tsumyrhel --zone=us-east1-b -- -NfL 6002:localhost:6002
# Start tensorboard with port number given in ssh ... this is port forwarding
	tensorboard --logdir="./pglogs/firstpgexp_fresh/train/" --port 6001 &
	tensorboard --logdir="./pglogs/firstpgexp_fresh/eval/" --port 6003 &
# Process to start training of the model in batch
* nohup python run_summarization.py --mode=train --data_path=/home/pnagula/cnn-dailymail/finished_files/chunked/train_* --vocab_path=/home/pnagula/cnn-dailymail/finished_files/vocab --log_root=./pglogs/ --exp_name=firstpgexp_fresh_400 --max_enc_steps=400  --vocab_size=90000  > plog_train_firstpgexp_fresh_400.log &

* After achieving a stagnation in validation loss curve or the error is increasing in validation loss curve stop the training and take the best model from validation dataset. run the training process one more time with --convert_to_coverage_model=True and --coverage=True this will initialize coverage variables in the best model and training stops. 

* nohup python run_summarization.py --mode=train --data_path=/home/pnagula/cnn-dailymail/finished_files/chunked/train_* --vocab_path=/home/pnagula/cnn-dailymail/finished_files/vocab --log_root=./pglogs/ --exp_name=firstpgexp_fresh_400 --max_enc_steps=400  --vocab_size=90000 --convert_to_coverage_model=True --coverage=True > plog_train_firstpgexp_fresh_400_coverage.log &

* Now run training process one more time with just --coverage=True for couple of thousands iterations and then stop the training process.

* nohup python run_summarization.py --mode=train --data_path=/home/pnagula/cnn-dailymail/finished_files/chunked/train_* --vocab_path=/home/pnagula/cnn-dailymail/finished_files/vocab --log_root=./pglogs/ --exp_name=firstpgexp_fresh_400 --max_enc_steps=400  --vocab_size=90000 --coverage=True > plog_train_firstpgexp_fresh_400_coverage.log &

