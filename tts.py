import pyttsx3

def speak(txt):
    engine = pyttsx3.init()
    engine.say(txt)
    engine.runAndWait()

speak("Surya ")