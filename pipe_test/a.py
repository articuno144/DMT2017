import win32pipe, win32file

p = win32pipe.CreateNamedPipe(r'\\.\pipe\test_pipe',
    win32pipe.PIPE_ACCESS_DUPLEX,
    win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_WAIT,
    1, 65536, 65536,300,None)

win32pipe.ConnectNamedPipe(p, None)


data = "Hello Pipe"  
win32file.WriteFile(p, data)
