import socket

print (f"Socket's original default timeout: {socket.getdefaulttimeout()}")

try:
    socket.setdefaulttimeout(100.0)
except socket.error as error:
    print (f"Error occured will trying to change the default timeout: {error}")

print (f"Socket's new default timeout: {socket.getdefaulttimeout()}")