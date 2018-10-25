# coding=utf-8
import sys
import os


def generate(models_path, model_paths):
    txt_paths = []
    for a_face in model_paths:
        face_path = os.path.join(models_path, a_face)
        model_images = sorted(os.listdir(face_path))
        model_images = [x for x in model_images if x.split('.')[-1] == 'pgm']  # delete file name that are not *.pgm
        a_txt_path = os.path.join(face_path, '1.txt')
        txt_paths.append(a_txt_path)
        with open(a_txt_path, 'w') as f:
            f.write('0\n')
            f.write('12\n')
            for i in xrange(0, 12, 1):
                f.write(os.path.join(face_path, model_images[i])+'\n')
    with open("/home/creator/Projects/DL/MVCNN-TensorFlow/data/view/test_lists.txt",'w') as f:
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