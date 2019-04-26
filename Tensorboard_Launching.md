# First create reverse tunnel enabled ssh connection 
	gcloud compute ssh pnagula@tsumyrhel --zone=us-east1-b -- -NfL 6002:localhost:6002
# Start tensorboard with port number given in ssh ... this is port forwarding
	tensorboard --logdir="./pglogs/firstpgexp_fresh/train/" --port 6001 &
	tensorboard --logdir="./pglogs/firstpgexp_fresh/eval/" --port 6003 &
