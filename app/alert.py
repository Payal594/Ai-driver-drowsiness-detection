import winsound

def play_alert():
    duration = 1000   # milliseconds
    freq = 1000       # Hz
    winsound.Beep(freq, duration)
