"""字幕產生器.vbs 的靜態檢查（Linux 上沒有 Windows Script Host 可以執行它，所以至少把最容易出錯的地方檢查掉）。

檢查：純 ASCII、只有 CRLF、Option Explicit 是第一個敘述、If／For／Do／Sub／Function／Select／With 有配對、
Exit For／Sub／Function 出現在對的地方、用到的名稱都有宣告（Option Explicit 下打錯字會在執行時才出錯）。
只支援這個檔案用到的寫法；不是通用的 VBScript 解析器。"""
import re

KEYWORDS = {w.lower() for w in """
If Then Else ElseIf End Sub Function For Each In To Step Next Exit Do Loop While Until Dim Const Set On Error Resume
GoTo Call Option Explicit Not And Or Xor Mod True False Nothing Empty Null Select Case With Is ByVal ByRef New Private
Public Class Property Get Let Wend""".split()}
BUILTINS = {w.lower() for w in """
Chr LCase UCase Trim Replace InStr Len Split UBound LBound Array Join MsgBox CreateObject WScript Err
vbCrLf vbCritical vbExclamation vbInformation vbTextCompare""".split()}


def strip_code(line: str) -> str:
    """去掉註解，字串內容換成空字串（只留引號），這樣字串裡的字不會被當成程式。"""
    out, in_string, i = [], False, 0
    while i < len(line):
        c = line[i]
        if in_string:
            if c == '"':
                if line[i + 1:i + 2] == '"':
                    i += 1                         # 字串裡的兩個引號 = 一個引號
                else:
                    in_string = False
                    out.append('"')
        elif c == '"':
            in_string = True
            out.append('"')
        elif c == "'":
            break                                  # 註解
        else:
            out.append(c)
        i += 1
    if in_string:
        raise ValueError(f"字串沒有結尾：{line!r}")
    return "".join(out).strip()


def logical_lines(text: str):
    """(行號, 程式碼)；行尾的 _ 把下一行接上來。"""
    result, pending, start = [], "", 0
    for number, raw in enumerate(text.split("\r\n"), start=1):
        code = strip_code(raw)
        if code.endswith(" _") or code == "_":
            pending, start = pending + code[:-1] + " ", start or number
            continue
        if pending:
            code, number, pending, start = pending + code, start, "", 0
        if code:
            result.append((number, code))
    if pending:
        raise ValueError("最後一行是續行符號 _")
    return result


def lint(text: str) -> list[str]:
    problems = []
    if any(ord(c) > 127 for c in text):
        problems.append("含有非 ASCII 字元（Windows Script Host 會把 .vbs 當成 ANSI 讀）")
    if re.search(r"(?<!\r)\n", text):
        problems.append("有單獨的 LF（應該全部是 CRLF）")
    try:
        lines = logical_lines(text)
    except ValueError as e:
        return problems + [str(e)]
    if not lines or lines[0][1].lower() != "option explicit":
        problems.append("第一個敘述不是 Option Explicit")

    stack, declared, procedures = [], set(), set()
    used = []                                       # (行號, 名稱)
    for number, code in lines:
        low = code.lower()
        if ":" in code.replace('""', ""):
            problems.append(f"第 {number} 行：用了「:」分隔多個敘述（這個檔案不用）")
        # ---- 區塊配對 ----
        m = re.match(r"(?:public |private )?(sub|function)\s+(\w+)\s*(?:\((.*)\))?$", low)
        if m:
            stack.append((m.group(1), number))
            procedures.add(m.group(2))
            declared.add(m.group(2))
            declared.update(x.strip().split()[-1] for x in (m.group(3) or "").split(",") if x.strip())
            continue
        if low.startswith("if ") and low.endswith(" then"):
            stack.append(("if", number))
        elif low.startswith("select case"):
            stack.append(("select", number))
        elif low.startswith("with "):
            stack.append(("with", number))
        elif low.startswith("for "):
            stack.append(("for", number))
        elif low.startswith("do") and re.match(r"do\b", low):
            stack.append(("do", number))
        else:
            closer = {"end if": "if", "end select": "select", "end with": "with", "end sub": "sub", "end function": "function",
                      "next": "for", "loop": "do"}
            key = re.match(r"(end (?:if|select|with|sub|function))\b|(next|loop)\b", low)
            if key:
                want = closer[key.group(0)]
                if not stack or stack[-1][0] != want:
                    problems.append(f"第 {number} 行：{key.group(0)} 沒有對應的開頭（目前開著的是 {stack[-1] if stack else '無'}）")
                else:
                    stack.pop()
            exit_kind = re.match(r"exit (for|do|sub|function)\b", low)
            if exit_kind and exit_kind.group(1) not in [s[0] for s in stack]:
                problems.append(f"第 {number} 行：{code} 不在對應的區塊裡")
        # ---- 宣告 ----
        if low.startswith("dim "):
            for part in code[4:].split(","):
                declared.add(re.match(r"\s*(\w+)", part).group(1).lower())
            continue
        if low.startswith("const "):
            declared.add(re.match(r"const\s+(\w+)", low).group(1))
        # ---- 用到的名稱（點後面的成員不算）----
        for ident in re.finditer(r"(?<![\w.])([A-Za-z_]\w*)", code):
            used.append((number, ident.group(1)))
    for kind, number in stack:
        problems.append(f"第 {number} 行開始的 {kind} 沒有結尾")
    known = declared | KEYWORDS | BUILTINS
    for number, ident in used:
        if ident.lower() not in known:
            problems.append(f"第 {number} 行：{ident} 沒有宣告（Option Explicit 下會在執行時出錯）")
    return sorted(set(problems))
