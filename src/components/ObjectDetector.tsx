import { useState, useEffect, useRef, useCallback } from 'react';
import { pipeline } from '@huggingface/transformers';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Camera, CameraOff, Volume2, VolumeX, Eye, Settings } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

interface Detection {
  label: string;
  score: number;
  box: {
    xmin: number;
    ymin: number;
    xmax: number;
    ymax: number;
  };
}

const ObjectDetector = () => {
  const [isDetecting, setIsDetecting] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [pipeline_, setPipeline] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [speechEnabled, setSpeechEnabled] = useState(true);
  const [confidence, setConfidence] = useState(0.5);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const lastSpokenRef = useRef<string>('');
  
  const { toast } = useToast();

  // Initialize the AI model
  const initializePipeline = useCallback(async () => {
    setIsLoading(true);
    try {
      toast({
        title: "Loading AI Model",
        description: "Initializing object detection model...",
      });
      
      const detector = await pipeline('object-detection', 'Xenova/detr-resnet-50', {
        device: 'webgpu',
        dtype: 'fp16',
      });
      
      setPipeline(detector);
      toast({
        title: "AI Model Ready",
        description: "Object detection is now available!",
        variant: "default",
      });
      
      // Announce model ready
      if (speechEnabled) {
        speak("BlindVision AI model is ready. You can now start object detection.");
      }
    } catch (error) {
      console.error('Error loading model:', error);
      toast({
        title: "Model Loading Failed", 
        description: "Failed to load AI model. Please check your connection.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  }, [speechEnabled, toast]);

  // Text-to-speech function
  const speak = useCallback((text: string) => {
    if (!speechEnabled || isSpeaking) return;
    
    // Avoid repeating the same announcement
    if (text === lastSpokenRef.current) return;
    lastSpokenRef.current = text;
    
    setIsSpeaking(true);
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.9;
    utterance.pitch = 1.1;
    utterance.volume = 1;
    
    utterance.onend = () => {
      setIsSpeaking(false);
      setTimeout(() => {
        lastSpokenRef.current = '';
      }, 2000);
    };
    
    utterance.onerror = () => {
      setIsSpeaking(false);
    };
    
    window.speechSynthesis.speak(utterance);
  }, [speechEnabled, isSpeaking]);

  // Start camera
  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: { ideal: 'environment' }
        }
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
      }
      
      toast({
        title: "Camera Started",
        description: "Video feed is now active",
      });
      
      speak("Camera started. You can now begin object detection.");
    } catch (error) {
      toast({
        title: "Camera Error",
        description: "Unable to access camera. Please check permissions.",
        variant: "destructive",
      });
      speak("Unable to access camera. Please check camera permissions.");
    }
  }, [toast, speak]);

  // Stop camera
  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    toast({
      title: "Camera Stopped",
      description: "Video feed has been stopped",
    });
    speak("Camera stopped.");
  }, [toast, speak]);

  // Object descriptions for detailed feedback
  const getObjectDescription = (label: string): string => {
    const descriptions: { [key: string]: string } = {
      'person': 'A person is in front of you. They appear to be standing or moving in the area.',
      'car': 'A car is detected nearby. Be aware of this vehicle in your surroundings.',
      'chair': 'A chair is visible. This is a piece of furniture you can sit on.',
      'table': 'A table is in the area. This is a flat surface, likely for placing items.',
      'bottle': 'A bottle is detected. This appears to be a container, possibly for drinks.',
      'cup': 'A cup or mug is visible. This is a drinking vessel, likely containing liquid.',
      'book': 'A book is detected in the area. This is reading material with pages.',
      'laptop': 'A laptop computer is visible. This is an electronic device for computing.',
      'phone': 'A mobile phone is detected. This is a communication device.',
      'dog': 'A dog is in the area. This is a friendly animal, likely a pet.',
      'cat': 'A cat is nearby. This is a small domestic animal, often kept as a pet.',
      'bicycle': 'A bicycle is detected. This is a two-wheeled vehicle for transportation.',
      'motorcycle': 'A motorcycle is visible. This is a motorized two-wheeled vehicle.',
      'bus': 'A bus is in the area. This is a large public transportation vehicle.',
      'truck': 'A truck is detected nearby. This is a large vehicle for carrying cargo.',
      'traffic light': 'A traffic light is visible. This controls traffic flow at intersections.',
      'stop sign': 'A stop sign is detected. This is a red octagonal traffic control sign.',
      'bench': 'A bench is in the area. This is outdoor seating, often found in parks.',
      'backpack': 'A backpack is visible. This is a bag worn on the back for carrying items.',
      'handbag': 'A handbag is detected. This is a bag typically carried by hand.',
      'suitcase': 'A suitcase is in the area. This is luggage for traveling.',
      'umbrella': 'An umbrella is visible. This is protection from rain or sun.',
      'tie': 'A necktie is detected. This is formal clothing worn around the neck.',
      'skis': 'Skis are visible. These are equipment for skiing on snow.',
      'snowboard': 'A snowboard is detected. This is equipment for snowboarding.',
      'sports ball': 'A sports ball is in the area. This is equipment used in various games.',
      'kite': 'A kite is visible. This is a flying object controlled by strings.',
      'baseball bat': 'A baseball bat is detected. This is sports equipment for hitting balls.',
      'baseball glove': 'A baseball glove is visible. This is protective sports equipment.',
      'skateboard': 'A skateboard is in the area. This is a board with wheels for riding.',
      'surfboard': 'A surfboard is detected. This is equipment for surfing on waves.',
      'tennis racket': 'A tennis racket is visible. This is equipment for playing tennis.',
      'wine glass': 'A wine glass is detected. This is a drinking vessel for wine.',
      'fork': 'A fork is visible. This is eating utensil with prongs.',
      'knife': 'A knife is detected. This is a cutting utensil with a sharp blade.',
      'spoon': 'A spoon is in the area. This is an eating utensil for liquids.',
      'bowl': 'A bowl is visible. This is a round container for food or liquids.',
      'banana': 'A banana is detected. This is a yellow curved fruit.',
      'apple': 'An apple is visible. This is a round fruit, often red or green.',
      'sandwich': 'A sandwich is in the area. This is food between slices of bread.',
      'orange': 'An orange is detected. This is a round citrus fruit.',
      'broccoli': 'Broccoli is visible. This is a green vegetable with tree-like florets.',
      'carrot': 'A carrot is detected. This is an orange root vegetable.',
      'pizza': 'Pizza is in the area. This is a round flatbread with toppings.',
      'donut': 'A donut is visible. This is a sweet ring-shaped pastry.',
      'cake': 'A cake is detected. This is a sweet baked dessert.',
      'couch': 'A couch or sofa is visible. This is large seating furniture.',
      'bed': 'A bed is detected. This is furniture for sleeping.',
      'dining table': 'A dining table is in the area. This is furniture for eating meals.',
      'toilet': 'A toilet is visible. This is bathroom sanitary equipment.',
      'tv': 'A television is detected. This is an electronic device for viewing programs.',
      'remote': 'A remote control is visible. This device controls electronic equipment.',
      'keyboard': 'A computer keyboard is detected. This is for typing on computers.',
      'mouse': 'A computer mouse is visible. This is for controlling computer cursors.',
      'microwave': 'A microwave oven is in the area. This appliance heats food quickly.',
      'oven': 'An oven is detected. This is a cooking appliance for baking.',
      'toaster': 'A toaster is visible. This appliance makes toast from bread.',
      'sink': 'A sink is in the area. This is for washing dishes or hands.',
      'refrigerator': 'A refrigerator is detected. This appliance keeps food cold.',
      'clock': 'A clock is visible. This device shows the current time.',
      'vase': 'A vase is detected. This is a container often used for flowers.',
      'teddy bear': 'A teddy bear is in the area. This is a soft stuffed toy.',
      'hair drier': 'A hair dryer is visible. This device dries and styles hair.',
      'toothbrush': 'A toothbrush is detected. This is for cleaning teeth.'
    };
    
    return descriptions[label] || `A ${label} is detected in your surroundings. This object is now visible in the camera view.`;
  };

  // Track previously announced objects to avoid repetition
  const previousDetectionsRef = useRef<Set<string>>(new Set());

  // Perform object detection
  const detectObjects = useCallback(async () => {
    if (!pipeline_ || !videoRef.current || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    const video = videoRef.current;

    if (!context) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0);

    try {
      const imageData = canvas.toDataURL('image/jpeg');
      const results = await pipeline_(imageData);
      
      const filteredResults = results.filter((detection: Detection) => 
        detection.score >= confidence
      );
      
      setDetections(filteredResults);

      if (filteredResults.length > 0 && speechEnabled) {
        // Get the highest confidence detection that hasn't been announced recently
        const newDetections = filteredResults.filter(detection => 
          !previousDetectionsRef.current.has(detection.label)
        );

        if (newDetections.length > 0) {
          // Announce the most confident new detection
          const topDetection = newDetections[0];
          const description = getObjectDescription(topDetection.label);
          speak(description);
          
          // Track this detection
          previousDetectionsRef.current.add(topDetection.label);
          
          // Clear tracking after 10 seconds to allow re-announcement
          setTimeout(() => {
            previousDetectionsRef.current.delete(topDetection.label);
          }, 10000);
        }
      } else if (filteredResults.length === 0) {
        // Clear previous detections when no objects are visible
        previousDetectionsRef.current.clear();
      }
    } catch (error) {
      console.error('Detection error:', error);
    }
  }, [pipeline_, confidence, speechEnabled, speak]);

  // Start/stop detection
  const toggleDetection = useCallback(async () => {
    if (!isDetecting) {
      if (!pipeline_) {
        await initializePipeline();
        return;
      }
      
      if (!streamRef.current) {
        await startCamera();
      }
      
      setIsDetecting(true);
      intervalRef.current = setInterval(detectObjects, 2000);
      speak("Object detection started. I will announce what I see around you.");
    } else {
      setIsDetecting(false);
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      speak("Object detection stopped.");
    }
  }, [isDetecting, pipeline_, initializePipeline, startCamera, detectObjects, speak]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      stopCamera();
    };
  }, [stopCamera]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (event: KeyboardEvent) => {
      if (event.code === 'Space') {
        event.preventDefault();
        toggleDetection();
      } else if (event.key === 'm' || event.key === 'M') {
        setSpeechEnabled(prev => !prev);
        speak(speechEnabled ? "Audio disabled" : "Audio enabled");
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [toggleDetection, speechEnabled, speak]);

  return (
    <div className="min-h-screen bg-background p-4 space-y-6">
      {/* Header */}
      <header className="text-center space-y-4 animate-fade-in-up">
        <div className="flex items-center justify-center gap-3">
          <Eye className="h-12 w-12 text-primary animate-pulse-glow" />
          <h1 className="text-4xl font-bold bg-gradient-primary bg-clip-text text-transparent">
            BlindVision
          </h1>
        </div>
        <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
          Real-time object detection with audio guidance for the visually impaired
        </p>
      </header>

      {/* Main Controls */}
      <div className="flex flex-col items-center space-y-6">
        {/* Video Feed */}
        <Card className="relative p-4 bg-card border-2">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="w-full max-w-2xl rounded-lg"
            style={{ aspectRatio: '4/3' }}
          />
          <canvas
            ref={canvasRef}
            className="hidden"
          />
          
          {/* Detection Overlay */}
          {detections.length > 0 && (
            <div className="absolute top-6 left-6 bg-black/80 text-white p-3 rounded-lg">
              <h3 className="font-bold text-success mb-2">Detected Objects:</h3>
              {detections.slice(0, 5).map((detection, index) => (
                <div key={index} className="text-sm">
                  {detection.label} ({Math.round(detection.score * 100)}%)
                </div>
              ))}
            </div>
          )}
        </Card>

        {/* Main Action Button */}
        <Button
          onClick={toggleDetection}
          disabled={isLoading}
          variant={isDetecting ? "destructive" : "vision"}
          size="xl"
          className="min-w-64"
          aria-label={isDetecting ? "Stop object detection" : "Start object detection"}
        >
          {isLoading ? (
            <>
              <Settings className="animate-spin" />
              Loading AI Model...
            </>
          ) : isDetecting ? (
            <>
              <CameraOff />
              Stop Detection
            </>
          ) : (
            <>
              <Camera />
              Start Detection
            </>
          )}
        </Button>

        {/* Control Panel */}
        <div className="flex flex-wrap gap-4 justify-center">
          <Button
            onClick={() => {
              setSpeechEnabled(prev => !prev);
              speak(speechEnabled ? "Audio disabled" : "Audio enabled");
            }}
            variant={speechEnabled ? "audio" : "outline"}
            size="lg"
            aria-label={speechEnabled ? "Disable audio" : "Enable audio"}
          >
            {speechEnabled ? <Volume2 /> : <VolumeX />}
            {speechEnabled ? 'Audio On' : 'Audio Off'}
          </Button>

          <Button
            onClick={startCamera}
            variant="outline"
            size="lg"
            disabled={!!streamRef.current}
            aria-label="Start camera"
          >
            <Camera />
            Start Camera
          </Button>

          <Button
            onClick={stopCamera}
            variant="outline" 
            size="lg"
            disabled={!streamRef.current}
            aria-label="Stop camera"
          >
            <CameraOff />
            Stop Camera
          </Button>
        </div>

        {/* Status Indicators */}
        <div className="flex gap-4 text-center">
          <div className={`p-3 rounded-lg ${isDetecting ? 'bg-success/20 text-success' : 'bg-muted text-muted-foreground'}`}>
            <div className="font-bold">Detection</div>
            <div className="text-sm">{isDetecting ? 'Active' : 'Inactive'}</div>
          </div>
          
          <div className={`p-3 rounded-lg ${speechEnabled ? 'bg-primary/20 text-primary' : 'bg-muted text-muted-foreground'}`}>
            <div className="font-bold">Audio</div>
            <div className="text-sm">{speechEnabled ? 'Enabled' : 'Disabled'}</div>
          </div>
          
          <div className={`p-3 rounded-lg ${streamRef.current ? 'bg-success/20 text-success' : 'bg-muted text-muted-foreground'}`}>
            <div className="font-bold">Camera</div>
            <div className="text-sm">{streamRef.current ? 'Active' : 'Inactive'}</div>
          </div>
        </div>

        {/* Instructions */}
        <Card className="p-6 max-w-2xl text-center space-y-3">
          <h3 className="text-xl font-bold text-primary">How to Use</h3>
          <div className="space-y-2 text-muted-foreground">
            <p><strong>Spacebar:</strong> Start/Stop object detection</p>
            <p><strong>M key:</strong> Toggle audio on/off</p>
            <p><strong>Voice Guide:</strong> Listen for real-time object descriptions</p>
            <p><strong>Tip:</strong> Point your camera at objects for best results</p>
          </div>
        </Card>
      </div>
    </div>
  );
};

export default ObjectDetector;