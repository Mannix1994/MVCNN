# coding=utf-8
import sys
import os


def generate(models_path, model_paths):
    txt_paths = []
    for a_face in model_paths:
        face_path = os.path.join(models_path, a_face)
        model_images = sorted(os.listdir(face_path))
        model_images = [x for x in model_images if x.split('.')[-1] == 'pgm']  # delete file name that are not *.pgm
        same_model_count = len(model_images) // 12
        for i in range(0, same_model_count):
            a_txt_path = os.path.join(face_path, '%d.txt' % i)
            txt_paths.append(a_txt_path)
            with open(a_txt_path, 'w') as f:
                f.write('0\n')
                f.write('12\n')
                for a_path in model_images[i * 12:(i + 1) * 12]:
                    f.write(os.path.join(face_path, a_path)+'\n')
    with open("/home/creator/Projects/DL/MVCNN-TensorFlow/data/view/train_lists.txt", 'w') as f:
        for index, a_txt_path in enumerate(txt_paths):
            f.write(a_txt_path+' %d\n' % index)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("python list.py model_path")
        exit(-1)
    else:
        models_path = sys.argv[1]
        model_paths = sorted(os.listdir(models_path))
        generate(models_path,model_paths)