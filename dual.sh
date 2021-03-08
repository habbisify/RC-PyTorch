RC_ROOT="/home/kapyla/Documents/LBLC"
DATE_RC="1109_1715"
DATE_QC="1115_1729"

DATASET_DIR="$RC_ROOT/datasets/train_oi_r_subset_clf"
TEMP_DIR="$RC_ROOT/datasets/train_oi_r_subset_clf_predicted_qs"
LOGS_DIR="$RC_ROOT/models"

mkdir "$TEMP_DIR"

pushd $RC_ROOT/RC-PyTorch/src

for i in {0..1}
do
	echo "Epoch: $i"
	
	# Correct q values for RC
	python -u get_optimal_qs.py \
		$DATASET_DIR \
		$LOGS_DIR/$DATE_QC

	echo "Images being copied..."
	python copy_imgs.py $DATASET_DIR $TEMP_DIR
	echo "Images copied"

	# Train RC
	CUDA_VISIBLE_DEVICES=0 python -u train.py \
	   	configs/ms/gdn_wide_deep3.cf \
 	   	configs/dl/new_oi_q12_14_128.cf \
 	   	$LOGS_DIR \		
 	   	-p unet_skip \
		--restore $DATE_RC \
		--restore_continue

	# Groundtruth for QC
	LIVE_HISTORY=1 CUDA_VISIBLE_DEVICES=0 python -u run_test.py \
    		"$LOGS_DIR" $DATE_RC "AUTOEXPAND:$DATASET_DIR/train_oi_r_subset_clf" \
    		--restore_itr 1000000 \
    		--qstrategy MIN

	# Train QC
	CUDA_VISIBLE_DEVICES=0 python -u train.py \
	   	configs/ms/clf/down2_nonorm_down.cf \
		configs/dl/clf/model1715.cf \
	   	$LOGS_DIR \
		--restore $DATE_QC \
		--restore_continue
done
