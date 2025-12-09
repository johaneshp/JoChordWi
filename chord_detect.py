import crema
import librosa

def get_chord_sequence(audio_file, audio_path):
    chord_labels=crema.models.chord.predict(audio_file)
    chords=[]
    for obs in chord_labels:
        chords.append({
           'start': obs.time,
           'end': obs.time +obs.duration,
           'chord': obs.value,
           'confidence' :obs.confidence 
        })

    return chords

 #TODO: implement chunking if more than one minute,
 #      save output to csv or json
 