"""測試用的替身：真的 tkinter，只是拖放相關的方法什麼都不做。
（真的 tkdnd 原生函式庫在沒有實體螢幕的環境會讓整個程序崩潰，而且拖放本來就沒辦法在自動測試裡模擬。）"""
try:
    import tkinter

    tkinter.Misc.drop_target_register = lambda self, *args, **kwargs: None
    tkinter.Misc.dnd_bind = lambda self, *args, **kwargs: None

    class TkinterDnD:
        Tk = tkinter.Tk
except ImportError:  # 沒有 tkinter：介面測試會自己跳過
    TkinterDnD = None

DND_FILES = "DND_Files"
