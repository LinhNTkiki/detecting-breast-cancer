# -*- coding: utf-8 -*-
"""AI_Kiểm tra cuối kỳ.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1wMfcP19Nuy7YkR6EqJNTHToU_eHgnt92
"""

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/

!kaggle datasets download -d aryashah2k/breast-ultrasound-images-dataset

import zipfile
zip_ref = zipfile.ZipFile('/content/breast-ultrasound-images-dataset.zip', 'r')
zip_ref.extractall('/content')
zip_ref.close()

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from PIL import Image
import shutil

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import random
from sklearn.model_selection import train_test_split
from torchvision.models import resnet50
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import classification_report, confusion_matrix

import seaborn as sns
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, ColorJitter

import glob

# Đường dẫn tới thư mục chứa dataset đã giải nén
image_directory = '/content/Dataset_BUSI_with_GT'

# Tìm tất cả các file ảnh .png trong thư mục và các thư mục con
breast_imgs = glob.glob(f'{image_directory}/**/*.png', recursive=True)

print(f'Tổng số file ảnh: {len(breast_imgs)}')
print('Một vài file ảnh đầu tiên:')
for img in breast_imgs[:5]:
    print(img)

# Hiển thị một vài file ảnh từ danh sách breast_imgs với kích thước cố định 6x6 inch bằng cách sử dụng thư viện PIL và Matplotlib

from PIL import Image
import matplotlib.pyplot as plt

# Thiết lập kích thước cố định cho ảnh
fixed_size = (6, 6)  # Đặt kích thước cố định là 6x6 inch

# Hiển thị một vài file ảnh
for img_path in breast_imgs[:5]:
    img = Image.open(img_path)

    # Tạo một hình ảnh với kích thước cố định
    plt.figure(figsize=fixed_size)

    plt.imshow(img)
    plt.axis('off')
    plt.show()

# Hiển thị các hình ảnh và mặt nạ (mask) từ ba thư mục tương ứng với các nhãn "benign", "malignant", và "normal" trong một hình gồm ba hàng và bốn cột,
#mỗi ảnh được điều chỉnh kích thước thành 200x200 pixel
# Định nghĩa các nhãn và thư mục tương ứng
labels = ['benign', 'malignant', 'normal']
data_dir = '/content/Dataset_BUSI_with_GT'
# Tạo 1 hình để hiển thị các ảnh
fig, axs = plt.subplots(3, 4, figsize=(18, 12))

# Đặt chiều rộng và chiều cao mong muốn cho mỗi ảnh
desired_width = 200
desired_height = 200

for i, label in enumerate(labels):
    label_dir = os.path.join(data_dir, label)

    # Lấy danh sách các file ảnh
    image_files = [file for file in os.listdir(label_dir) if file.endswith('.png')]

    # Sắp xếp danh sách các file ảnh
    image_files = sorted(image_files)
    # Loop 4 file ảnh đầu tiên
    for j in range(4):
        if j < len(image_files):
            # Tải và thay đổi kích thước ảnh
            image_path = os.path.join(label_dir, image_files[j])
            image = Image.open(image_path)
            image = image.resize((desired_width, desired_height), Image.ANTIALIAS)

            # Xác định nhãn dựa trên ảnh hay mặt nạ
            if j % 2 == 0:
                image_label = f'{label} - Image {j // 2 + 1}'
            else:
                image_label = f'{label} - Image {j // 2 + 1} Mask'

            # Hiển thị ảnh với nhãn tương ứng
            axs[i, j].imshow(image)
            axs[i, j].set_title(image_label)
            axs[i, j].axis('off')

plt.tight_layout()
plt.show()

# Thực hiện việc liệt kê các đường dẫn tới các file ảnh trong các thư mục tương ứng với các nhãn "benign", "malignant", và "normal" trong dataset "Dataset_BUSI_with_GT".
# Các đường dẫn tới các file ảnh được lưu trữ vào các danh sách benign_imgs, malignant_imgs, và normal_imgs tương ứng.
import os

# Định nghĩa các nhãn và thư mục tương ứng
labels = ['benign', 'malignant', 'normal']
data_dir = '/content/Dataset_BUSI_with_GT'

# Khởi tạo các danh sách trống để lưu trữ các đường dẫn tới ảnh
benign_imgs = []
malignant_imgs = []
normal_imgs = []

# Lặp qua từng nhãn và lấy các ảnh tương ứng
for label in labels:
    label_dir = os.path.join(data_dir, label)

    # Lấy danh sách các file ảnh
    image_files = [file for file in os.listdir(label_dir) if file.endswith('.png')]

    # Sắp xếp danh sách các file ảnh
    image_files = sorted(image_files)

    # Thêm các đường dẫn ảnh vào các danh sách tương ứng
    for img in image_files:
        image_path = os.path.join(label_dir, img)
        if label == 'benign':
            benign_imgs.append(image_path)
        elif label == 'malignant':
            malignant_imgs.append(image_path)
        elif label == 'normal':
            normal_imgs.append(image_path)

# In ra số lượng ảnh của mỗi loại
benign_num = len(benign_imgs)
malignant_num = len(malignant_imgs)
normal_num = len(normal_imgs)

total_img_num = benign_num + malignant_num + normal_num

print('Số lượng ảnh siêu âm có tình trạng ung thư lành tính: {}'.format(benign_num))
print('Số lượng ảnh siêu âm có tình trạng ung thư ác tính: {}'.format(malignant_num))
print('Số lượng ảnh siêu âm có tình trạng bình thường: {}'.format(normal_num))
print('Tổng số lượng ảnh: {}'.format(total_img_num))

import pandas as pd
import plotly.express as px

# Định nghĩa DataFrame
data_insight_1 = pd.DataFrame({'Tình trạng bệnh' : ['benign','malignant', 'normal'],'Số lượng bệnh nhân' : [891,421,266]})

# Tạo biểu đồ bar
bar = px.bar(data_frame=data_insight_1, x='Tình trạng bệnh', y='Số lượng bệnh nhân', color='Tình trạng bệnh')

# Cập nhật layout của biểu đồ
bar.update_layout(title_text='Biểu đồ thể hiện số lượng ảnh bệnh theo tình trạng', title_x=0.5)

# Hiển thị biểu đồ
bar.show()

"""Xây dựng và huấn huyện mô hình"""

# Định nghĩa hàm huấn luyện mô hình với cơ chế dừng sớm (early stopping) để ngăn chặn overfitting

#Khởi tạo các danh sách để lưu trữ giá trị loss
train_losses = [] #lưu trữ giá trị loss trong quá trình huấn luyện
val_losses = []  #Lưu trữ giá trị loss trong quá trình kiểm tra

#Hàm huấn luyện mô hình với early stopping trong nhiều epoch
#Thiết lập và khởi tạo
def train_model_with_early_stopping(model, lossFunction, optimizer, scheduler, dataloaders, dataset_sizes, class_names, device, num_epochs=20, patience=2):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')  # Initialize best_loss with a large value
    consecutive_epochs_without_improvement = 0
    #Vòng lặp epoch chạy qua các epoch và in ra số epoch hiện tại
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Lặp qua dữ liệu
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Đặt lại gradient của các tham số
                optimizer.zero_grad()

                #Lan truyền tiến
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = lossFunction(outputs, labels)

                    # Lan truyền ngược + tối ưu chỉ khi ở giai đoạn huấn luyện
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Thêm giá trị loss của giai đoạn huấn luyện vào
                if phase == 'train':
                    train_losses.append(loss.item())
                else:
                    val_losses.append(loss.item())

                # Thống kê
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # Early stopping: Kiểm tra nếu loss trên tập kiểm tra đã cải thiện
            if phase == 'validation':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    consecutive_epochs_without_improvement = 0
                else:
                    consecutive_epochs_without_improvement += 1


                val_losses.append(epoch_loss)

            # Kiểm tra nếu tiêu chí dừng sớm được thỏa mãn
        if consecutive_epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch} epochs")
            break

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:.4f}'.format(best_loss))

    # Tải lại trọng số mô hình tốt nhất
    model.load_state_dict(best_model_wts)

    # Tính toán báo cáo phân loại và ma trận nhầm lẫn cho dữ liệu kiểm tra
    y_true = []
    y_pred = []

    # Đặt mô hình vào chế độ đánh giá
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloaders['validation']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    #Đánh giá mô hình trên tập kiểm tra
    # Generate classification report
    target_names = [str(class_names[i]) for i in range(len(class_names))]
    print(classification_report(y_true, y_pred, target_names=target_names))

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    return model

# Thiết lập thiết bị để đảm bảo rằng mô hình sẽ chạy trên GPU nếu có, còn nếu không sẽ chạy trên CPU.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Thực hiện quá trình gắn kết các ảnh với mặt nạ tương ứng và lưu kết quả vào thư mục đầu ra
import warnings

# Tắt các cảnh báo DeprecationWarning và ResourceWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)

# Đặt đường dẫn đến thư mục đầu vào chứa ảnh và mặt nạ
input_dir = '/content/Dataset_BUSI_with_GT'

# Đặt đường dẫn đến thư mục đầu ra để lưu trữ các ảnh được gắn kết
output_dir = '/kaggle/working/OverlayedImages'

# Tạo các thư mục con cho mỗi nhãn
labels = ['benign', 'malignant', 'normal']
for label in labels:
    os.makedirs(os.path.join(output_dir, label), exist_ok=True)

# Hàm để gắn kết các ảnh và mặt nạ, thay đổi kích thước nếu cần và lưu kết quả
def overlay_and_save(image_path, mask_path, output_path):
    try:
        # Kiểm tra xem cả tệp ảnh và mặt nạ có tồn tại không
        if os.path.exists(image_path) and os.path.exists(mask_path):
            # Mở ảnh và mặt nạ tương ứng
            image = Image.open(image_path)
            mask = Image.open(mask_path)

            # Đảm bảo cả hai ảnh có cùng chế độ màu
            if image.mode != mask.mode:
                mask = mask.convert(image.mode)

            # Thay đổi kích thước ảnh nếu kích thước không khớp
            if image.size != mask.size:
                image = image.resize(mask.size)

            # Gắn kết ảnh với mặt nạ
            overlayed = Image.blend(image, mask, alpha=0.5)

            # Lưu ảnh đã gắn kết vào thư mục nhãn tương ứng
            label = os.path.basename(os.path.dirname(image_path))
            output_path = os.path.join(output_dir, label, os.path.basename(image_path))
            overlayed.save(output_path)
        else:
            #print(f"File not found for: {image_path} or {mask_path}. Skipping...")
            pass
    except Exception as e:
        print(f"An error occurred for: {image_path} or {mask_path}. Error: {str(e)}")

# Lặp qua các thư mục con (benign, malignant, normal)
for label in labels:
    label_dir = os.path.join(input_dir, label)
    if os.path.isdir(label_dir):
        for image_filename in os.listdir(label_dir):
            if image_filename.endswith('.png'):
                image_path = os.path.join(label_dir, image_filename)
                # Tạo đường dẫn cho tệp mặt nạ dựa trên quy ước đặt tên
                mask_filename = image_filename.replace('.png', '_mask.png')
                mask_path = os.path.join(label_dir, mask_filename)
                overlay_and_save(image_path, mask_path, output_dir)

print("Overlayed images have been saved to /kaggle/working/OverlayedImages directory.")

# Đếm số lượng tệp tin trong thư mục đầu vào và thư mục đầu ra trước và sau khi thực hiện việc gắn kết ảnh với mặt nạ
import os

# Định nghĩa một hàm count_files_in_directory(directory) để đếm số lượng tệp tin trong một thư mục
def count_files_in_directory(directory):
    return sum(len(files) for _, _, files in os.walk(directory))

# Đặt đường dẫn đến thư mục đầu vào và thư mục đầu ra.
input_dir = '/content/Dataset_BUSI_with_GT'
output_dir = '/kaggle/working/OverlayedImages'


input_counts = {}
output_counts = {}

# Đếm tệp tin trong thư mục đầu vào
for label in os.listdir(input_dir):
    label_dir = os.path.join(input_dir, label)
    if os.path.isdir(label_dir):
        input_counts[label] = count_files_in_directory(label_dir)

# Đếm tệp tin trong thư mục đầu ra
for label in os.listdir(output_dir):
    label_dir = os.path.join(output_dir, label)
    if os.path.isdir(label_dir):
        output_counts[label] = count_files_in_directory(label_dir)

# In ra số lượng tệp tin
print("File Counts Before Overlay-includes masks:")
for label, count in input_counts.items():
    print(f"{label}: {count} files")

print("\nFile Counts After Overlay:")
for label, count in output_counts.items():
    print(f"{label}: {count} files")

# Tạo ra một biểu đồ hiển thị các hình ảnh gắn mặt nạ được lưu trong thư mục overlayed_dir. Nó chia thành 3 hàng và 4 cột, mỗi hàng tương ứng với một nhãn (benign, malignant, normal), và mỗi cột hiển thị một hình ảnh
# Đặt đường dẫn đến thư mục chứa hình ảnh đã gắn mặt nạ
overlayed_dir = '/kaggle/working/OverlayedImages'

# Tạo các thư mục con cho mỗi nhãn
labels = ['benign', 'malignant', 'normal']
label_dirs = [os.path.join(overlayed_dir, label) for label in labels]

# Tạo một hình để hiển thị các hình ảnh
fig, axs = plt.subplots(3, 4, figsize=(20, 15))

# Đặt chiều rộng và chiều cao mong muốn cho mỗi hình ảnh
desired_width = 800  # Adjust as needed
desired_height = 800  # Adjust as needed

# Lặp qua mỗi nhãn và hiển thị 4 hình ảnh đầu tiên
for i, label_dir in enumerate(label_dirs):
    # Lấy danh sách các tệp hình ảnh và sắp xếp chúng
    images = [image for image in os.listdir(label_dir) if image.endswith('.png')]
    images.sort(key=lambda x: int(x.split('(')[1].split(')')[0]))  # Sort the images by number in parentheses

    for j, image_filename in enumerate(images[:4]):  # Display the first 4 images
        image_path = os.path.join(label_dir, image_filename)
        image = Image.open(image_path)


        image = image.resize((desired_width, desired_height), Image.ANTIALIAS)

        # Display the image in the subplot
        axs[i, j].imshow(image)
        axs[i, j].set_title(f'{labels[i]} Image {j + 1}')
        axs[i, j].axis('off')

plt.show()

# Hiển thị 3 loại ảnh siêu âm(actual, mask, overlayed) của lớp benign
# Định nghĩa các đường dẫn tới thư mục chứa dataset và thư mục chứa hình ảnh overlayed
input_dir = '/content/Dataset_BUSI_with_GT'
overlayed_dir = '/kaggle/working/OverlayedImages/benign'

# Tạo một figure với 3 ô để hiển thị hình ảnh.
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Đặt kích thước mong muốn cho hiển thị là 300x300 pixels.
desired_width = 300
desired_height = 300

# Chọn 1 file ảnh cụ thể
image_filename = 'benign (10).png'

# Xây dựng đường dẫn file cho ảnh gốc, ảnh mask và ảnh overlayed.
actual_image_path = os.path.join(input_dir, 'benign', image_filename)
mask_image_path = os.path.join(input_dir, 'benign', image_filename.replace(".png", "_mask.png"))
overlayed_image_path = os.path.join(overlayed_dir, image_filename)

# Tải các ảnh từ các đường dẫn đã xây dựng.
actual_image = Image.open(actual_image_path)
mask_image = Image.open(mask_image_path)
overlayed_image = Image.open(overlayed_image_path)

# Thay đổi kích thước các ảnh theo kích thước mong muốn.
actual_image = actual_image.resize((desired_width, desired_height), Image.ANTIALIAS)
mask_image = mask_image.resize((desired_width, desired_height), Image.ANTIALIAS)
overlayed_image = overlayed_image.resize((desired_width, desired_height), Image.ANTIALIAS)

# Hiển thị các ảnh trên figure với tiêu đề tương ứng cho từng ảnh
axs[0].imshow(actual_image)
axs[0].set_title('benign -Actual Image')
axs[0].axis('off')

axs[1].imshow(mask_image, cmap='gray')
axs[1].set_title('benign - Mask')
axs[1].axis('off')

axs[2].imshow(overlayed_image)
axs[2].set_title('benign - Overlayed Image')
axs[2].axis('off')

plt.tight_layout()
plt.show()

# Cân bằng lại dữ liệu bằng cách áp dụng các biến đổi dữ liệu nâng cao cho các lớp thiểu số trong quá trình huấn luyện mô hình
# Định nghĩa các lớp thiểu số
class_names = ['malignant', 'normal','benign']
minority_classes = ['malignant', 'normal']

# Định nghĩa các biến đổi tùy chỉnh cho các lớp thiểu số(Tạo một chuỗi các biến đổi dữ liệu (augmentation) bao gồm lật ngang ngẫu nhiên, xoay ngẫu nhiên và thay đổi màu sắc ngẫu nhiên.)
minority_class_transforms = transforms.Compose([
    RandomHorizontalFlip(p=0.9),  # Apply with 90% probability
    RandomRotation(15, expand=False, center=None),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
])

# Định nghĩa các biến đổi dữ liệu cho tập huấn luyện, kiểm tra và kiểm định
# Với tập huấn luyện: Áp dụng các biến đổi cơ bản như thay đổi kích thước, cắt tâm, chuyển đổi sang tensor và chuẩn hóa
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        # Ngoài ra, nếu lớp dữ liệu thuộc lớp thiểu số, sẽ áp dụng thêm các biến đổi tùy chỉnh với xác suất 50%
        transforms.RandomApply([minority_class_transforms], p=0.5) if any(cls in minority_classes for cls in class_names) else transforms.RandomApply([], p=0.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),

    #Với tập kiểm tra và kiểm định: Chỉ áp dụng các biến đổi cơ bản như thay đổi kích thước, cắt tâm, chuyển đổi sang tensor và chuẩn hóa, không áp dụng các biến đổi tùy chỉnh.
    'validation': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Chuẩn bị cho quá trình huấn luyện mô hình bằng cách tải và sắp xếp các hình ảnh thành các tập huấn luyện, kiểm định và kiểm tra, đồng thời sao chép các hình ảnh vào các thư mục tương ứng
# Đặt đường dẫn đến thư mục chứa dữ liệu đầu vào: data_dir là thư mục chứa các hình ảnh đã được overlayed
data_dir = '/kaggle/working/OverlayedImages'

# Tạo danh sách để lưu trữ đường dẫn file và nhãn: file_paths (lưu trữ đường dẫn tới các file ảnh, labels lưu trữ các nhãn tương ứng)
file_paths = []
labels = []

# Duyệt qua các thư mục con (benign, malignant, normal)
for label in os.listdir(data_dir):
    label_dir = os.path.join(data_dir, label)
    if os.path.isdir(label_dir):
        for image_file in os.listdir(label_dir):
            if image_file.endswith('.png') and not (image_file.endswith('_mask.png') or
                                                     image_file.endswith('_mask_1.png') or
                                                     image_file.endswith('_mask_2.png')):
                image_path = os.path.join(label_dir, image_file)
                labels.append(label)
                file_paths.append(image_path)

# Tạo một DataFrame để lưu trữ các đường dẫn file và nhãn: data chứa hai cột là Image_Path và Label.
data = pd.DataFrame({'Image_Path': file_paths, 'Label': labels})

# Chia bộ dữ liệu thành các tập huấn luyện, kiểm tra và kiểm định
train_data, test_data = train_test_split(data, test_size=0.15, random_state=42, stratify=data['Label'])
train_data, val_data = train_test_split(train_data, test_size=0.15, random_state=42, stratify=train_data['Label'])

# Định nghĩa các đường dẫn cho các thư mục huấn luyện, kiểm định và kiểm tra: train_dir, val_dir, test_dir.
train_dir = '/kaggle/working/train'
val_dir = '/kaggle/working/validation'
test_dir = '/kaggle/working/test'

# Tạo các thư mục huấn luyện, kiểm định và kiểm tra, cùng các thư mục con tương ứng với các nhãn
for label in labels:
    os.makedirs(os.path.join(train_dir, label), exist_ok=True)
    os.makedirs(os.path.join(val_dir, label), exist_ok=True)
    os.makedirs(os.path.join(test_dir, label), exist_ok=True)

# Sao chép các hình ảnh vào các thư mục tương ứng
for _, row in train_data.iterrows():
    image_path = row['Image_Path']
    label = row['Label']
    shutil.copy(image_path, os.path.join(train_dir, label))

for _, row in val_data.iterrows():
    image_path = row['Image_Path']
    label = row['Label']
    shutil.copy(image_path, os.path.join(val_dir, label))

for _, row in test_data.iterrows():
    image_path = row['Image_Path']
    label = row['Label']
    shutil.copy(image_path, os.path.join(test_dir, label))

# Đếm và in ra số lượng file trong các thư mục con tương ứng với các lớp (benign, malignant, normal) trong các tập dữ liệu huấn luyện, kiểm định và kiểm tra
import os

# Đặt đường dẫn tới thư mục huấn luyện: train_dir chứa các hình ảnh của tập huấn luyện.
train_dir = '/kaggle/working/train'

# Liệt kê các thư mục con (benign, malignant, normal)
subdirectories = ['benign', 'malignant', 'normal']

# Tạo một từ điển để lưu trữ số lượng file
file_counts = {}

# Duyệt qua các thư mục con và đếm số lượng file trong từng thư mục
for subdirectory in subdirectories:
    subdirectory_path = os.path.join(train_dir, subdirectory)
    if os.path.exists(subdirectory_path):
        file_count = len(os.listdir(subdirectory_path))
        file_counts[subdirectory] = file_count

# In số lượng file trong các thư mục con của tập huấn luyện
for category, count in file_counts.items():
    print(f"Train {category}: {count}")


#Lặp lại các bước trên cho thư mục kiểm định: validation_dir chứa các hình ảnh của tập kiểm định
validation_dir = '/kaggle/working/validation'
subdirectories = ['benign', 'malignant', 'normal']
file_counts = {}
for subdirectory in subdirectories:
    subdirectory_path = os.path.join(validation_dir, subdirectory)
    if os.path.exists(subdirectory_path):
        file_count = len(os.listdir(subdirectory_path))
        file_counts[subdirectory] = file_count

for category, count in file_counts.items():
    print(f"Validation {category}: {count}")


#Lặp lại các bước trên cho thư mục kiểm tra: test_dir chứa các hình ảnh của tập kiểm tra
test_dir = '/kaggle/working/test'
subdirectories = ['benign', 'malignant', 'normal']
file_counts = {}
for subdirectory in subdirectories:
    subdirectory_path = os.path.join(test_dir, subdirectory)
    if os.path.exists(subdirectory_path):
        file_count = len(os.listdir(subdirectory_path))
        file_counts[subdirectory] = file_count
for category, count in file_counts.items():
    print(f"test {category}: {count}")

# Chuẩn bị dữ liệu cho việc huấn luyện mô hình bằng PyTorch

# Đặt đường dẫn đến thư mục chứa dữ liệu
data_dir='/kaggle/working/'

# Tạo các bộ dữ liệu cho huấn luyện, kiểm định và kiểm tra
image_datasets = {
    x: ImageFolder(
        root=os.path.join(data_dir, x),
        transform=data_transforms[x]
    )
    for x in ['train', 'validation', 'test']
}

# Xác định kích thước batch cho DataLoader
batch_size = 32

# Tạo DataLoader cho huấn luyện, kiểm định và kiểm tra
dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
               for x in ['train', 'validation', 'test']}

# Tính toán kích thước của từng bộ dữ liệu
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation', 'test']}

# Lấy nhãn lớp
class_names = image_datasets['train'].classes

# In ra kích thước của các bộ dữ liệu và nhãn lớp
print("Dataset Sizes:", dataset_sizes)
print("Class Labels:", class_names)

# Chuyển đổi mô hình ResNet-50 đã được huấn luyện trước (pre-trained) sang mô hình fine-tuned cho một tác vụ phân loại ảnh mới bằng cách thay thế lớp fully connected cuối cùng

import torch
import torchvision.models as models
# Tải mô hình ResNet-50 đã được huấn luyện trước
resnet50_model= resnet50(pretrained=True)
# In ra thông tin mô hình ResNet-50
print(resnet50)
# Cho phép tất cả các tham số của mô hình được tối ưu hóa
for param in resnet50_model.parameters():
    param.requires_grad = True

# Lấy số lượng đầu vào của lớp fully connected cuối cùng của ResNet
# bởi vì chúng ta sẽ thay thế nó bằng lớp fully connected mới.
in_features = resnet50_model.fc.in_features

# Thay đổi lớp fully connected cuối cùng của ResNet được huấn luyện trước.
resnet50_model.fc = nn.Linear(in_features, len(class_names))

num_hidden_layers = sum(1 for _, layer in resnet50_model.named_modules() if isinstance(layer, torch.nn.Conv2d))

print("Số lớp ẩn trong mạng CNN ResNet-50:", num_hidden_layers)
# In ra thông tin của các layer trong mạng CNN
for name, layer in resnet50_model.named_modules():
    if isinstance(layer, torch.nn.Conv2d):
        print(f"Layer name: {name}, Output size: {layer.weight.size()}")

pip install torchviz

import torch
import torchvision.models as models
from torchviz import make_dot

# Tạo mô hình ResNet-50 đã được huấn luyện trước
resnet50_model = models.resnet50(pretrained=True)

# Chuyển mô hình vào chế độ đánh giá
resnet50_model.eval()

# Tạo tensor đầu vào giả lập với kích thước tương ứng
x = torch.randn(1, 3, 224, 224)

# Tạo đồ thị mô hình bao gồm cả lớp đầu vào
y = resnet50_model(x)
dot = make_dot(y, params=dict(resnet50_model.named_parameters()))

# Lưu đồ thị vào tệp và hiển thị nó
dot.format = 'png'
dot.render('resnet50_graph')

# Hiển thị hình ảnh đồ thị mô hình
from IPython.display import Image
Image('resnet50_graph.png')

# Định nghĩa optimzation algorithm
optimizer = optim.Adam(resnet50_model.parameters(), lr=0.00005)

# Decay LR by a factor of 0.1 every 7 epochs
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
# Định nghĩa loss functions
Loss_Function = nn.CrossEntropyLoss()

# Thực hiện quá trình huấn luyện mô hình, kiểm soát và lưu trữ mô hình đã được huấn luyện vào một tệp

# Xác định đường dẫn và tên tệp lưu trữ
save_dir = r"C:/Users/User"
save_path = os.path.join(save_dir, "resnet50_fineTuning.h5")

# Tạo thư mục lưu trữ nếu nó không tồn tại
os.makedirs(save_dir, exist_ok=True)
# Di chuyển mô hình vào thiết bị được chỉ định(CPU)
resnet50_fineTuning = resnet50_model.to(device)
# Huấn luyện mô hình với early-stopping
model_fineTuning = train_model_with_early_stopping(
    resnet50_fineTuning, Loss_Function, optimizer,scheduler,
    dataloaders, dataset_sizes, class_names, device,num_epochs=20, patience=2)
# Lưu trữ mô hình đã được huấn luyện
torch.save(model_fineTuning, "C:/Users/User/resnet50_fineTuning.h5")

"""ĐÁNH GIÁ MÔ HÌNH"""

# Đánh giá hiệu suất của mô hình fine-tuning trên dữ liệu kiểm tra chưa nhìn thấy trước đó
# Định nghĩa tên nhãn
label_names = [str(class_names[i]) for i in range(len(class_names))]

# Tính toán báo cáo phân loại và ma trận nhầm lẫn trên dữ liệu kiểm tra
y_true = []
y_pred = []
# Đặt mô hình ở chế độ đánh giá
model_fineTuning.eval()

with torch.no_grad():
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model_fineTuning(inputs)
        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Tạo báo cáo phân loại
classification_rep = classification_report(y_true, y_pred, target_names=label_names, output_dict=True)

# Tạo ma trận nhầm lẫn
confusion_mat = confusion_matrix(y_true, y_pred)

# Vẽ biểu đồ ma trận nhầm lẫn
plt.figure(figsize=(5, 3))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=label_names, yticklabels=label_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Tính toán ma trận nhầm lẫn dưới dạng phần trăm và vẽ nó dưới dạng biểu đồ heatmap để trực quan hóa hiệu suất phân loại của mô hình
confusion_mtx_percent = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis] * 100

f, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(confusion_mtx_percent, annot=True, linewidths=0.01, cmap="BuPu", linecolor="gray", fmt='.1f', ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Percentage)")
plt.show()

# Tạo một biểu đồ heatmap từ báo cáo phân loại được tính toán trước đó để trực quan hóa các chỉ số đánh giá hiệu suất của mô hình trên từng lớp dữ liệu
plt.figure(figsize=(6, 4))
sns.heatmap(pd.DataFrame(classification_rep).iloc[:-1, :].T, annot=True, cmap='Blues', fmt='.2f')  # Simplify classification report
plt.title('Classification Report Heatmap')
plt.show()

# In ra các chỉ số đánh giá hiệu suất của mô hình
print("Simplified Classification Report:")
print(pd.DataFrame(classification_rep).iloc[:-1, :])

from sklearn.metrics import classification_report
print(classification_report(y_true,y_pred))

# vẽ biểu đồ để trực quan hóa sự thay đổi của hàm mất mát (loss) trên tập huấn luyện và tập validation qua các epoch trong quá trình huấn luyện mô hình
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

"""KIỂM THỬ MÔ HÌNH ĐÃ HUẤN LUYỆN"""

comparison_df = pd.DataFrame({ 'Actual': y_true,'Predicted': y_pred})

print(comparison_df[:25])

import numpy as np
import matplotlib.pyplot as plt

# Set the number of images to display
num_images_to_display = 15

# Tạo DataLoader cho tập dữ liệu kiểm tra
test_dataloader = DataLoader(image_datasets['test'], batch_size=num_images_to_display, shuffle=True, num_workers=4)

# Lấy một batch dữ liệu từ DataLoader
inputs, labels = next(iter(test_dataloader))

# Chuyển inputs lên thiết bị
inputs = inputs.to(device)

# Chuyển đổi hình ảnh thành ảnh xám
grayscale_images = inputs.cpu().numpy().mean(axis=1)  # Convert RGB to grayscale

# Dự đoán nhãn của các hình ảnh
with torch.no_grad():
    model_fineTuning.eval()
    outputs = model_fineTuning(inputs)
    _, preds = torch.max(outputs, 1)

# Vẽ hình ảnh xám với nhãn và dự đoán
plt.figure(figsize=(15, 20))
for i in range(num_images_to_display):
    ax = plt.subplot(5, 3, i + 1)
    ax.axis('off')
    ax.set_title(f'Actual: {class_names[labels[i]]}\nPredicted: {class_names[preds[i]]}')
    plt.imshow(grayscale_images[i], cmap='gray')

plt.show()

