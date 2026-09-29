' Double-click this file to open "Whisper Subtitle Generator" - no black cmd window will appear.
' Keep this .vbs in the same folder as w1_1.py, align_worker.py and the whisper_app folder.
'
' Python is looked up in this order:
'   1. pythonw on PATH (default when Python is installed with "Add to PATH")
'   2. pyw (the Python Launcher for Windows, installed by default from python.org)
' If you use a specific Python / virtual environment, put its full path to pythonw.exe
' in PYTHONW_PATH below, e.g. "C:\Users\me\venvs\whisper\Scripts\pythonw.exe".
'
' This file is intentionally ASCII-only: Windows Script Host reads .vbs files as ANSI,
' so Chinese text saved as UTF-8 here would show up garbled.

Option Explicit

Const PYTHONW_PATH = ""
Const APP_TITLE = "Whisper Subtitle Generator"

Dim objShell, objFSO, strScriptDir, strMainScript, strWorker, strPackage, strArgs
Dim launchers, launcher, launched

Set objShell = CreateObject("WScript.Shell")
Set objFSO = CreateObject("Scripting.FileSystemObject")
strScriptDir = objFSO.GetParentFolderName(WScript.ScriptFullName)
strMainScript = objFSO.BuildPath(strScriptDir, "w1_1.py")
strWorker = objFSO.BuildPath(strScriptDir, "align_worker.py")
strPackage = objFSO.BuildPath(strScriptDir, "whisper_app\__init__.py")

If Not objFSO.FileExists(strMainScript) Then
    MsgBox "Cannot find w1_1.py in:" & vbCrLf & strScriptDir & vbCrLf & vbCrLf & _
           "Put this .vbs file in the same folder as w1_1.py.", vbCritical, APP_TITLE
    WScript.Quit 1
End If

If Not objFSO.FileExists(strPackage) Then
    MsgBox "The whisper_app folder is missing from:" & vbCrLf & strScriptDir & vbCrLf & vbCrLf & _
           "Put the whole whisper_app folder in the same folder as w1_1.py.", vbCritical, APP_TITLE
    WScript.Quit 1
End If

If Not objFSO.FileExists(strWorker) Then
    MsgBox "align_worker.py is missing from:" & vbCrLf & strScriptDir & vbCrLf & vbCrLf & _
           "Transcription still works, but timestamp alignment will fail." & vbCrLf & _
           "Put align_worker.py in the same folder as w1_1.py.", vbExclamation, APP_TITLE
End If

objShell.CurrentDirectory = strScriptDir
strArgs = " " & Chr(34) & strMainScript & Chr(34)

If PYTHONW_PATH <> "" Then
    launchers = Array(Chr(34) & PYTHONW_PATH & Chr(34))
Else
    launchers = Array("pythonw", "pyw")
End If

launched = False
For Each launcher In launchers
    On Error Resume Next
    Err.Clear
    objShell.Run launcher & strArgs, 0, False
    If Err.Number = 0 Then launched = True
    On Error GoTo 0
    If launched Then Exit For
Next

If Not launched Then
    MsgBox "Could not start Python." & vbCrLf & vbCrLf & _
           "Tried: " & Join(launchers, ", ") & vbCrLf & vbCrLf & _
           "Make sure Python is installed and pythonw is on PATH," & vbCrLf & _
           "or set PYTHONW_PATH at the top of this .vbs file.", vbCritical, APP_TITLE
    WScript.Quit 1
End If
