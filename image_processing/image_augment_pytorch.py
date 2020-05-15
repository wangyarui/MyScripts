'''torch vison:1.4'''
import cv2
from PIL import Image
from torchvision import transforms

image_transforms = transforms.Compose([
        # transforms.Resize((96,96), interpolation=2),
        # transforms.Grayscale(),
        transforms.ToTensor(),
        # transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])

transform_toimage = transforms.Compose([
    transforms.ToPILImage()
])

if __name__ == '__main__':
    src_image  = cv2.imread("./images/test_2.jpg")

    img = Image.fromarray(src_image)
    # if self.transform is not None:
    img = image_transforms(img)

    show_img = transform_toimage(img)
    show_img.show()


