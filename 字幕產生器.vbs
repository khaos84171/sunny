' Double-click this file to open "Whisper Subtitle Generator" - no black cmd window will appear.
' Keep this .vbs in the same folder as w1_1.py, align_worker.py and the whisper_app folder.
'
' Which Python is used - every candidate is test-run first, the first one that works and is
' new enough (3.9 or newer) wins:
'   1. PYTHONW_PATH below, if you set it (a specific Python / virtual environment), e.g.
'      "C:\Users\me\venvs\whisper\Scripts\pythonw.exe". Nothing else is tried in that case.
'   2. Otherwise: pythonw.exe on PATH, then pyw.exe (the Python Launcher for Windows), then the
'      Python installs registered in the Windows registry (works even without "Add to PATH").
'   3. Last: the Microsoft Store shortcut in WindowsApps. It only works when Python was really
'      installed from the Store, which is why nothing is trusted before it has been test-run.
'
' The program is started hidden and this script waits for it to end. If it ends with an error
' code, a message box tells you where to look (whisper_app.log next to this file).
'
' This file is intentionally ASCII-only: Windows Script Host reads .vbs files as ANSI,
' so Chinese text saved as UTF-8 here would show up garbled.

Option Explicit

Const PYTHONW_PATH = ""
Const APP_TITLE = "Whisper Subtitle Generator"
Const MIN_MAJOR = 3
Const MIN_MINOR = 9

Dim objShell, objFSO, strScriptDir, strMainScript, strWorker, strPackage
Dim candidates(63), candidateCount
Dim chosenExe, chosenArgs, foundOld, i, parts, status, exitCode

Set objShell = CreateObject("WScript.Shell")
Set objFSO = CreateObject("Scripting.FileSystemObject")
strScriptDir = objFSO.GetParentFolderName(WScript.ScriptFullName)
strMainScript = objFSO.BuildPath(strScriptDir, "w1_1.py")
strWorker = objFSO.BuildPath(strScriptDir, "align_worker.py")
strPackage = objFSO.BuildPath(strScriptDir, "whisper_app\__init__.py")
candidateCount = 0

Function Quote(text)
    Quote = Chr(34) & text & Chr(34)
End Function

' Remember one way to start Python ("exe path|extra arguments"), ignoring duplicates.
Sub AddCandidate(exePath, extraArgs)
    Dim k, entry
    entry = exePath & "|" & extraArgs
    For k = 0 To candidateCount - 1
        If LCase(candidates(k)) = LCase(entry) Then Exit Sub
    Next
    If candidateCount <= UBound(candidates) Then
        candidates(candidateCount) = entry
        candidateCount = candidateCount + 1
    End If
End Sub

' Add every exeName found in a folder on PATH. The Microsoft Store shortcut folder
' (WindowsApps) is only added when storeAliasOnly is True, and left out otherwise.
Sub AddFromPath(exeName, extraArgs, storeAliasOnly)
    Dim pathValue, dirs, k, dirName, isStoreDir, fullPath
    On Error Resume Next
    pathValue = objShell.ExpandEnvironmentStrings(objShell.Environment("PROCESS").Item("PATH"))
    dirs = Split(pathValue, ";")
    For k = 0 To UBound(dirs)
        dirName = Trim(Replace(dirs(k), Chr(34), ""))
        If Len(dirName) > 0 Then
            isStoreDir = (InStr(1, dirName, "\WindowsApps", vbTextCompare) > 0)
            If isStoreDir = storeAliasOnly Then
                fullPath = objFSO.BuildPath(dirName, exeName)
                If objFSO.FileExists(fullPath) Then AddCandidate fullPath, extraArgs
            End If
        End If
    Next
End Sub

' Add the Python installs that the Python installer registered in the Windows registry.
Sub AddFromRegistry()
    Dim roots, suffixes, r, s, minor, installDir, fullPath
    roots = Array("HKCU\Software\Python\PythonCore\", "HKLM\Software\Python\PythonCore\", "HKLM\Software\WOW6432Node\Python\PythonCore\")
    suffixes = Array("", "-32")
    On Error Resume Next
    For Each r In roots
        For minor = 14 To MIN_MINOR Step -1
            For Each s In suffixes
                installDir = ""
                Err.Clear
                installDir = objShell.RegRead(r & MIN_MAJOR & "." & minor & s & "\InstallPath\")
                If Err.Number = 0 Then
                    If Len(installDir) > 0 Then
                        fullPath = objFSO.BuildPath(installDir, "pythonw.exe")
                        If objFSO.FileExists(fullPath) Then AddCandidate fullPath, ""
                    End If
                End If
            Next
        Next
    Next
End Sub

' Test-run one candidate. Returns 0 = works and is new enough, 1 = works but too old,
' 2 = does not run at all (missing, broken, or a Store shortcut without a real Python behind it).
Function ProbePython(exePath, extraArgs)
    Dim code, cmd, rc
    code = "import sys; sys.exit(0 if sys.version_info >= (" & MIN_MAJOR & ", " & MIN_MINOR & ") else 3)"
    cmd = Quote(exePath) & extraArgs & " -c " & Quote(code)
    On Error Resume Next
    rc = objShell.Run(cmd, 0, True)
    If Err.Number <> 0 Then
        ProbePython = 2
    ElseIf rc = 0 Then
        ProbePython = 0
    ElseIf rc = 3 Then
        ProbePython = 1
    Else
        ProbePython = 2
    End If
End Function

' ---------------------------------------------------------------- check the files

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

' ---------------------------------------------------------------- find a working Python

If PYTHONW_PATH <> "" Then
    AddCandidate PYTHONW_PATH, ""
Else
    AddFromPath "pythonw.exe", "", False
    AddFromPath "pyw.exe", " -3", False
    AddFromRegistry
    AddFromPath "pythonw.exe", "", True
End If

chosenExe = ""
chosenArgs = ""
foundOld = False
For i = 0 To candidateCount - 1
    parts = Split(candidates(i), "|")
    status = ProbePython(parts(0), parts(1))
    If status = 0 Then
        chosenExe = parts(0)
        chosenArgs = parts(1)
        Exit For
    ElseIf status = 1 Then
        foundOld = True
    End If
Next

If chosenExe = "" Then
    If foundOld Then
        MsgBox "Found Python, but it is older than " & MIN_MAJOR & "." & MIN_MINOR & "." & vbCrLf & vbCrLf & _
               "Please install Python " & MIN_MAJOR & "." & MIN_MINOR & " or newer from python.org" & vbCrLf & _
               "(tick ""Add python.exe to PATH"" in the installer).", vbCritical, APP_TITLE
    ElseIf PYTHONW_PATH <> "" Then
        MsgBox "PYTHONW_PATH does not point to a working Python:" & vbCrLf & PYTHONW_PATH & vbCrLf & vbCrLf & _
               "Fix the path at the top of this .vbs file, or empty it to let this script look for Python.", vbCritical, APP_TITLE
    Else
        MsgBox "Could not find a working Python " & MIN_MAJOR & "." & MIN_MINOR & " or newer." & vbCrLf & vbCrLf & _
               "Install Python from python.org (tick ""Add python.exe to PATH"")," & vbCrLf & _
               "or set PYTHONW_PATH at the top of this .vbs file to your pythonw.exe.", vbCritical, APP_TITLE
    End If
    WScript.Quit 1
End If

' ---------------------------------------------------------------- run the program

objShell.CurrentDirectory = strScriptDir
exitCode = 0
On Error Resume Next
exitCode = objShell.Run(Quote(chosenExe) & chosenArgs & " " & Quote(strMainScript), 0, True)
If Err.Number <> 0 Then
    MsgBox "Could not start the program with:" & vbCrLf & chosenExe & vbCrLf & vbCrLf & Err.Description, vbCritical, APP_TITLE
    WScript.Quit 1
End If
On Error GoTo 0

If exitCode <> 0 Then
    MsgBox "The program stopped with an error (exit code " & exitCode & ")." & vbCrLf & vbCrLf & _
           "Details are usually in these files in " & strScriptDir & ":" & vbCrLf & _
           "  whisper_app.log" & vbCrLf & _
           "  whisper_startup_error.txt", vbExclamation, APP_TITLE
End If
