$desktop = [Environment]::GetFolderPath('Desktop')

# Eski PersonalAI dosyalarini sil
Remove-Item "$desktop\PersonalAI*" -Force -ErrorAction SilentlyContinue

# Yeni QuantumTree.bat kopyala
Copy-Item "C:\Projects\quantumtree\QuantumTree.bat" -Destination $desktop -Force

Write-Host "Tamam! Masaustunde QuantumTree.bat olusturuldu."
