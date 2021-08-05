from PIL import Image

def my_dtree(feature1, feature2):
    f_name = './tree_imgs_png/' + min(feature1, feature2)  + '_' + max(feature1, feature2) +  '.png'
    image = Image.open(f_name)
    
    return image