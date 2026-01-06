$desktop = [Environment]::GetFolderPath('Desktop')

# VBS ve LNK sil, sadece BAT kalsın
Remove-Item "$desktop\PersonalAI.vbs" -Force -ErrorAction SilentlyContinue
Remove-Item "$desktop\PersonalAI.lnk" -Force -ErrorAction SilentlyContinue

Write-Host "Temizlendi! Sadece PersonalAI.bat kaldi."
