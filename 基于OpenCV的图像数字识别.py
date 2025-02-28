import cv2
import numpy as np
from collections import defaultdict

#使用opencv进行颜色聚类
def get_dominant_colors(img, k=2):
    pixels = img.reshape((-1, 3)).astype(np.float32)
    
    #定义k-means参数
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    #按出现频率排序颜色
    counts = defaultdict(int)
    for label in labels.flatten():
        counts[label] += 1
    sorted_colors = sorted(centers, key=lambda x: -counts[np.where(centers == x)[0][0]])
    
    return np.array(sorted_colors, dtype=np.uint8)

#生成二进制特征码
def image_to_pattern(img_path, grid_size=(5,3), threshold=30):
    #读取并预处理图像
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (300, 200))  
    
    #获取主色
    colors = get_dominant_colors(img)
    bg_color, digit_color = colors[0], colors[1]
    
    #创建颜色掩模
    color_diff = cv2.absdiff(img, digit_color)
    color_dist = cv2.norm(color_diff, axis=2)
    _, mask = cv2.threshold(color_dist.astype(np.uint8), threshold, 255, cv2.THRESH_BINARY_INV)
    
    #网格划分
    rows, cols = grid_size
    h, w = img.shape[:2]
    grid_h, grid_w = h//rows, w//cols
    
    pattern = []
    for i in range(rows):
        for j in range(cols):
            #计算每个网格的激活状态
            y1, y2 = i*grid_h, (i+1)*grid_h
            x1, x2 = j*grid_w, (j+1)*grid_w
            grid = mask[y1:y2, x1:x2]
            
            active_ratio = cv2.countNonZero(grid) / (grid.size + 1e-6)
            pattern.append('1' if active_ratio > 0.2 else '0')
    
    return ''.join(pattern)

class DigitRecognizer:
    def __init__(self, template_dir, grid_size=(5,3)):
        self.templates = self.load_templates(template_dir, grid_size)
    #加载模板库
    def load_templates(self, template_dir, grid_size):
        templates = {}
        for label in range(10):
            img_path = f"{template_dir}/{label}.png"
            img = cv2.imread(img_path)
            if img is not None:
                pattern = image_to_pattern(img_path, grid_size)
                templates[label] = pattern
        return templates
    
    #识别数字
    def recognize(self, img_path, confidence_thresh=0.7):
        input_pattern = image_to_pattern(img_path, grid_size=next(iter(self.templates.values())).shape)
        
        best_match = {'label': None, 'confidence': 0}
        for label, tpl_pattern in self.templates.items():
            match = sum(c1 == c2 for c1, c2 in zip(input_pattern, tpl_pattern)) / len(input_pattern)
            if match > best_match['confidence']:
                best_match = {'label': label, 'confidence': match}
        
        if best_match['confidence'] >= confidence_thresh:
            return best_match
        return {'label': None, 'confidence': 0}

#使用
if __name__ == "__main__":
    recognizer = DigitRecognizer("digit_templates")
    
    test_img = "test_image.png"
    result = recognizer.recognize(test_img)
    
    if result['label'] is not None:
        print(f"识别结果: 数字{result['label']} (置信度: {result['confidence']:.1%})")
    else:
        print("未能识别有效数字")