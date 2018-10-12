# coding=utf-8
import os
import sys


def write_train_model_path_to_txt(model_path='.', model_id=0):
    model_images = sorted(os.listdir(model_path))
    model_images = [x for x in model_images if x.split('.')[-1] == 'jpg']  # delete file name that are not *.jpg
    same_model_count = len(model_images) // 12
    for i in range(0, same_model_count):
        with open(os.path.join(model_path, str(i + 1) + '.txt'), mode='w') as f:
            f.write(str(model_id) + '\n')
            f.write(str(12) + '\n')
            for a_path in model_images[i * 12:(i + 1) * 12]:
                f.write(os.path.join(model_path, str(a_path) + '\n'))
    with open("/home/creator/Projects/DL/MVCNN-TensorFlow/data/view/train_lists.txt", mode='a') as list_file:
        for i in range(0, same_model_count):
            list_file.write(os.path.join(model_path, str(i + 1) + '.txt') + ' %d\n' % model_id)


def write_test_model_path_to_txt(model_path='.', model_id=0):
    model_images = sorted(os.listdir(model_path))
    model_images = [x for x in model_images if x.split('.')[-1] == 'jpg']  # delete file name that are not *.jpg
    same_model_count = len(model_images) // 12
    for i in range(0, same_model_count):
        with open(os.path.join(model_path, str(i + 1) + '.txt'), mode='w') as f:
            f.write(str(model_id) + '\n')
            f.write(str(12) + '\n')
            for a_path in model_images[i * 12:(i + 1) * 12]:
                f.write(os.path.join(model_path, str(a_path) + '\n'))
    with open("/home/creator/Projects/DL/MVCNN-TensorFlow/data/view/val_lists.txt", mode='a') as list_file:
        for i in range(0, same_model_count):
            list_file.write(os.path.join(model_path, str(i + 1) + '.txt') + ' %d\n' % model_id)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("python generate_data_list.py model_path")
        exit(-1)
    else:
        models_path = sys.argv[1]
        model_paths = sorted(os.listdir(models_path))
        for single_model_id, single_model_path in enumerate(model_paths):
            write_train_model_path_to_txt(models_path+'/'+single_model_path+'/train', single_model_id)
            write_test_model_path_to_txt(models_path+'/'+single_model_path+'/test', single_model_id)
