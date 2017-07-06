#original code form http://pjreddie.com/projects/mnist-in-csv/

#you can run script convert.py or download .csv files from the link above directly

#The format is:
#label, pix-11, pix-12, pix-13, ...


import os

def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()

path= os.getcwd()
head,tail= os.path.split(path)
path= head

convert(os.path.join(os.path.join(path,"Original dataset"),"train-images.idx3-ubyte"),
		os.path.join(os.path.join(path,"Original dataset"),"train-labels.idx1-ubyte"),
        "mnist_train.csv", 60000)
convert(os.path.join(os.path.join(path,"Original dataset"),"t10k-images.idx3-ubyte"),
		os.path.join(os.path.join(path,"Original dataset"),"t10k-labels.idx1-ubyte"),
        "mnist_test.csv", 10000)