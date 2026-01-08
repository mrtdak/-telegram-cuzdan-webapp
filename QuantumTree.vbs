Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "C:\Projects\quantumtree"
WshShell.Run "cmd /c python desktop_chat.py", 0, False
