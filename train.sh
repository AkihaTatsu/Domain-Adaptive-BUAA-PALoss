clear
rm -rf data/cache/
rm -rf data/other/annotations_cache_
rm -rf data/other/results
python code/tasks/train.py --dataset voc_2007_trainval --t_dataset clipart_train --cfg configs/baselines/voc2007_clipart.yaml --bs 1 --model oicr_lambda_log_distillation
# python code/tasks/train.py --dataset voc_2007_trainval --t_dataset comic_train --cfg configs/baselines/voc2007_comic.yaml --bs 1 --model oicr_lambda_log_distillation
# python code/tasks/train.py --dataset voc_2007_trainval --t_dataset watercolor_train --cfg configs/baselines/voc2007_watercolor.yaml --bs 1 --model oicr_lambda_log_distillation