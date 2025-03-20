import torch
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torchvision import transforms
from openpyxl import load_workbook
from pre_processing import (
    MaxResize, outputs_to_objects, visualize_detected_tables,
    objects_to_crops, apply_ocr, get_cell_coordinates_by_row,
    plot_results, TableTransformerForObjectDetection, AutoModelForObjectDetection,
    Predictor, Cfg, fig2img, save_ocr_results_to_excel
)


# Load inference model table detection
model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection", revision="no_timm")

# Load local model
# model = AutoModelForObjectDetection.from_pretrained("./local_model/table-transformer-detection/")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# image processing
file_path = 'images/bang2demo.png'
image = Image.open(file_path).convert("RGB")
width, height = image.size
resized_image = image.resize((int(0.6 * width), int(0.6 * height)))
#resized_image.save("./files/table_cropped.png")

detection_transform = transforms.Compose([
    MaxResize(800),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
pixel_values = detection_transform(image).unsqueeze(0)
pixel_values = pixel_values.to(device)

# Predict bounding box and label for tables in image
with torch.no_grad():
    outputs = model(pixel_values)

# Post-processing: convert outputs to objects
id2label = model.config.id2label
id2label[len(model.config.id2label)] = "no object"
objects = outputs_to_objects(outputs, image.size, id2label)

# Hiển thị các bảng đã được phát hiện
fig = visualize_detected_tables(image, objects)
plt.show()
visualized_image = fig2img(fig)

# Cắt các bảng từ ảnh
tokens = []
detection_class_thresholds = {
    "table": 0.5,
    "table rotated": 0.5,
    "no object": 10
}
crop_padding = 8
tables_crops = objects_to_crops(image, tokens, objects, detection_class_thresholds, padding=crop_padding)
cropped_table = tables_crops[0]['image'].convert("RGB")
plt.imshow(cropped_table)
plt.axis('off')
plt.show()
cropped_table.save("./files/table_cropped.png")

# Load inference model table structure recognition
structure_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-structure-recognition-v1.1-all")
# structure_model = TableTransformerForObjectDetection.from_pretrained("./local_model/table-structure-recognition-v1.1-all/")
structure_model.to(device)

# Áp dụng các bước tiền xử lý cho ảnh cắt được từ bảng
structure_transform = transforms.Compose([
    MaxResize(1000),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
pixel_values = structure_transform(cropped_table).unsqueeze(0)
pixel_values = pixel_values.to(device)

# Dự đoán các thành phần trong bảng (hàng, cột)
with torch.no_grad():
    outputs = structure_model(pixel_values)
structure_id2label = structure_model.config.id2label
structure_id2label[len(structure_id2label)] = "no object"
cells = outputs_to_objects(outputs, cropped_table.size, structure_id2label)
print("Cấu trúc của một ô:")
print(cells[0])  # In ra cấu trúc của ô đầu tiên
#canonicalization

# Hiển thị các ô được nhận diện trong bảng
cropped_table_visualized = cropped_table.copy()
draw = ImageDraw.Draw(cropped_table_visualized)
for cell in cells:
    draw.rectangle(cell["bbox"], outline="red")
plt.imshow(cropped_table_visualized)
plt.axis('off')
plt.show()

plot_results(cells, structure_model,cropped_table,class_to_visualize="table row" )
plot_results(cells, structure_model,cropped_table,class_to_visualize="table column" )
# Load inference model
# cfg = Cfg.load_config_from_name('vgg_transformer')
# cfg['weights'] = 'https://vocr.vn/data/vietocr/vgg_transformer.pth'

# Load local model
cfg = Cfg.load_config_from_file('./config/config_after_trainer.yml')
cfg['weights'] = '/kaggle/input/aaaaa/other/default/1/pretrain/vgg_transformer.pth'
cfg['cnn']['pretrained'] = True
cfg['device'] = 'cuda'
predictor = Predictor(cfg)

# Áp dụng OCR để nhận diện văn bản trong các ô của bảng
cell_coordinates = get_cell_coordinates_by_row(cells)
data = apply_ocr(cell_coordinates, cropped_table, predictor)
for row, row_data in data.items():
    print(row_data)

save_ocr_results_to_excel(data)

