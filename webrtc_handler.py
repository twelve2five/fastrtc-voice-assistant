import asyncio
import json
import logging
import os
import ssl
import uuid
from typing import Dict, Optional, Callable

import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCIceCandidate
from aiortc.contrib.media import MediaBlackhole, MediaRelay

logger = logging.getLogger("webrtc_handler")
pcs = set()
relay = MediaRelay()

class AudioTransformTrack(MediaStreamTrack):
    """
    A track that processes audio and sends it to a callback function
    """
    kind = "audio"

    def __init__(self, track, callback):
        super().__init__()
        self.track = track
        self.callback = callback

    async def recv(self):
        frame = await self.track.recv()
        # Process audio frame
        if self.callback:
            self.callback(frame)
        return frame

async def handle_offer(offer, audio_callback=None):
    offer_data = RTCSessionDescription(sdp=offer["sdp"], type=offer["type"])
    
    pc = RTCPeerConnection()
    pcs.add(pc)
    
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state is {pc.connectionState}")
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)
    
    @pc.on("track")
    def on_track(track):
        logger.info(f"Track {track.kind} received")
        if track.kind == "audio":
            pc.addTrack(AudioTransformTrack(relay.subscribe(track), audio_callback))
        
        @track.on("ended")
        async def on_ended():
            logger.info(f"Track {track.kind} ended")
    
    # Handle the incoming offer
    await pc.setRemoteDescription(offer_data)
    
    # Create an answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    
    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    }

async def add_ice_candidate(candidate, pc):
    if candidate and pc:
        candidate_data = RTCIceCandidate(
            sdpMLineIndex=candidate.get("sdpMLineIndex"),
            sdpMid=candidate.get("sdpMid"),
            candidate=candidate.get("candidate")
        )
        await pc.addIceCandidate(candidate_data) 