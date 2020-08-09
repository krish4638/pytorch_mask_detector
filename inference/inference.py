import torch
from torchvision import transforms
from PIL import Image


model = torch.hub.load_state_dict_from_url('https://vamshi4638.s3.amazonaws.com/mask_model_resnet101.pth', map_location='cpu')
class_names = ['with_mask', 'without_mask']


def process_image(image):
    print('[INFO] preprocess')
    pil_image = Image.open(image)
    image_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = image_transforms(pil_image)
    return img


def classify_face(image):
    print("[INFO] classification")
    img = process_image(image)
    img = img.unsqueeze_(0)
    img = img.float()
    model.eval()
    output = model(img)
    _, predicted = torch.max(output, 1)
    classification1 = predicted.data[0]
    index = int(classification1)
    print('[INFO] label: ', class_names[index])
    return class_names[index]

if __name__ == '__main__':
    image_path = '/home/mwebware/Documents/vamshi/pytorch_mask_detector/experiements/data/with_mask/0-with-mask.jpg'
    label = classify_face(image_path)
    print("the label is", label)









