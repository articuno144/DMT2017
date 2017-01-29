import win32pipe, win32file
import win32file
fileHandle = win32file.CreateFile("\\\\.\\pipe\\test_pipe",
                              win32file.GENERIC_READ | win32file.GENERIC_WRITE,
                              0, None,
                              win32file.OPEN_EXISTING,
                              0, None)
data = win32file.ReadFile(fileHandle, 4096)
print data
