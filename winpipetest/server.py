import time
import sys
import win32pipe, win32file, pywintypes
import struct


pipeName = "MessageServer"
bufferSize = 64 * 5


def main():
    print("pipe server")
    count = 0
    pipe = win32pipe.CreateNamedPipe(
        r'\\.\pipe\\' + pipeName,
        win32pipe.PIPE_ACCESS_DUPLEX,
        win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_READMODE_MESSAGE | win32pipe.PIPE_WAIT,
        1, bufferSize, bufferSize,
        0,
        None)
    try:
        print("waiting for client")
        win32pipe.ConnectNamedPipe(pipe, None)
        print("got client")

        while count < 10:
            print(f"writing message {count}")
            # convert to bytes

            some_data = str.encode(f"{count}")
            win32file.WriteFile(pipe, struct.pack("HddH", 10, 0.2, 0.003, 7))
            time.sleep(1)
            count += 1

        print("finished now")
    finally:
        win32file.CloseHandle(pipe)


if __name__ == '__main__':
    main()

