$WshShell = New-Object -ComObject WScript.Shell
$Desktop = [Environment]::GetFolderPath('Desktop')
$Shortcut = $WshShell.CreateShortcut("$Desktop\PersonalAI.lnk")
$Shortcut.TargetPath = "C:\Projects\quantumtree\PersonalAI.vbs"
$Shortcut.WorkingDirectory = "C:\Projects\quantumtree"
$Shortcut.Description = "PersonalAI Desktop Chat"
$Shortcut.Save()
Write-Host "Masaustu kisayolu olusturuldu!"
