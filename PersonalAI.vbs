Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "C:\Projects\quantumtree"
WshShell.Run "python desktop_chat.py", 0, False
