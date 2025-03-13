let pc = null;
let localStream = null;
let audioSender = null;

async function setupWebRTC() {
    // Create WebRTC peer connection
    pc = new RTCPeerConnection({
        iceServers: [
            { urls: 'stun:stun.l.google.com:19302' }
        ]
    });
    
    // Get local media stream
    localStream = await navigator.mediaDevices.getUserMedia({
        audio: true,
        video: false
    });
    
    // Add tracks to peer connection
    localStream.getTracks().forEach(track => {
        audioSender = pc.addTrack(track, localStream);
    });
    
    // Create offer
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    
    // Send offer to server (would need to be implemented)
    sendOfferToServer(pc.localDescription);
    
    // Set up event listeners for ICE candidates
    pc.onicecandidate = event => {
        if (event.candidate) {
            sendIceCandidateToServer(event.candidate);
        }
    };
    
    // Handle incoming tracks (audio responses)
    pc.ontrack = event => {
        const audioElement = document.getElementById('ai-response-audio');
        if (audioElement) {
            audioElement.srcObject = new MediaStream([event.track]);
        }
    };
}

function sendOfferToServer(offer) {
    // Send the offer to your backend
    // Implementation would depend on your server setup
    fetch('/webrtc/offer', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(offer)
    })
    .then(response => response.json())
    .then(answer => {
        pc.setRemoteDescription(new RTCSessionDescription(answer));
    })
    .catch(error => console.error('Error sending offer:', error));
}

function sendIceCandidateToServer(candidate) {
    // Send ICE candidate to server
    fetch('/webrtc/ice-candidate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(candidate)
    })
    .catch(error => console.error('Error sending ICE candidate:', error));
}

function startRecording() {
    // Unmute the audio track
    if (localStream) {
        localStream.getAudioTracks().forEach(track => {
            track.enabled = true;
        });
    }
}

function stopRecording() {
    // Mute the audio track
    if (localStream) {
        localStream.getAudioTracks().forEach(track => {
            track.enabled = false;
        });
    }
} 