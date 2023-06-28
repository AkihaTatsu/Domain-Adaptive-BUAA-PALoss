clear
rm -rf data/cache/
python code/tasks/test.py --dataset clipart_test --cfg configs/baselines/voc2007_clipart.yaml --model oicr_lambda_log_distillation --load_ckpt snapshots/oicr_lambda_log_distillation/2023-05-21_00-08-12/ckpt/model_step149999.pth
# python code/tasks/train.py --dataset voc_2007_trainval --t_dataset comic_train --cfg configs/baselines/voc2007_comic.yaml --bs 1 --model oicr_lambda_log_distillation
# python code/tasks/train.py --dataset voc_2007_trainval --t_dataset watercolor_train --cfg configs/baselines/voc2007_watercolor.yaml --bs 1 --model oicr_lambda_log_distillation