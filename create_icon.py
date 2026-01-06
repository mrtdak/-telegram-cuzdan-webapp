"""Basit ağaç ikonu oluştur"""
from PIL import Image, ImageDraw

# 64x64 ikon
size = 64
img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# Ağaç gövdesi (kahverengi)
trunk_color = (139, 90, 43)
draw.rectangle([26, 40, 38, 60], fill=trunk_color)

# Ağaç yaprakları (yeşil daireler)
leaf_color = (34, 139, 34)
draw.ellipse([12, 8, 52, 45], fill=leaf_color)  # Ana yaprak
draw.ellipse([8, 18, 36, 42], fill=leaf_color)  # Sol yaprak
draw.ellipse([28, 18, 56, 42], fill=leaf_color)  # Sağ yaprak
draw.ellipse([18, 5, 46, 32], fill=leaf_color)  # Üst yaprak

# Parıltı efekti (açık yeşil)
highlight = (50, 205, 50)
draw.ellipse([22, 12, 38, 28], fill=highlight)

# ICO olarak kaydet
img.save('C:/Projects/quantumtree/tree_icon.ico', format='ICO', sizes=[(64, 64)])
print("✅ tree_icon.ico oluşturuldu!")
