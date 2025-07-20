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
      'person': 'A person is detected in your view. This is a human being who may be walking, standing, or moving around in your immediate area.',
      'bicycle': 'A bicycle is detected nearby. This is a two-wheeled pedal-powered vehicle used for transportation, exercise, or recreation.',
      'car': 'A car is present in the scene. This is a four-wheeled motor vehicle designed for passenger transportation on roads.',
      'motorcycle': 'A motorcycle is visible. This is a two-wheeled motor vehicle that provides faster transportation than bicycles but requires more skill to operate.',
      'airplane': 'An airplane is visible in the sky above. This is a powered aircraft with wings that flies through the air for transportation.',
      'bus': 'A bus is nearby. This is a large motor vehicle designed to carry many passengers, often used for public transportation.',
      'train': 'A train is detected. This is a series of connected rail cars that travel on tracks, used for transporting passengers or cargo over long distances.',
      'truck': 'A truck is present. This is a large motor vehicle designed primarily for transporting goods and materials rather than passengers.',
      'boat': 'A boat is visible on the water. This is a watercraft used for traveling across rivers, lakes, or oceans for transportation or recreation.',
      'traffic light': 'A traffic light is ahead of you. This is a signaling device that uses colored lights - red, yellow, and green - to control traffic flow at intersections.',
      'fire hydrant': 'A fire hydrant is located nearby. This is a water supply point that firefighters connect their hoses to when fighting fires.',
      'stop sign': 'A stop sign is visible. This is a red, octagonal traffic sign that requires all vehicles to come to a complete stop before proceeding.',
      'parking meter': 'A parking meter is present. This is a device that collects payment for parking in designated spaces, usually found on city streets.',
      'bench': 'A bench is in your vicinity. This is outdoor seating furniture, typically made of wood or metal, found in parks, bus stops, or public spaces.',
      'bird': 'A bird is detected in the area. This is a feathered animal with wings that can fly, and it is part of the local wildlife around you.',
      'cat': 'A cat is present nearby. This is a small, domesticated feline animal often kept as a pet, known for its independence and agility.',
      'dog': 'A dog is detected in the area. This is a domesticated canine animal, commonly kept as a companion pet and known for loyalty to humans.',
      'horse': 'A horse is visible. This is a large, four-legged mammal historically used for riding, pulling carts, and farm work.',
      'sheep': 'A sheep is in the scene. This is a woolly farm animal raised primarily for its wool, meat, and sometimes milk.',
      'cow': 'A cow is present. This is a large farm animal, a bovine mammal raised for milk production, meat, and leather.',
      'elephant': 'An elephant is detected. This is the largest land mammal, with a long trunk, large ears, and tusks, native to Africa and Asia.',
      'bear': 'A bear is visible - exercise caution. This is a large, powerful mammal that can be dangerous. Keep your distance and avoid sudden movements.',
      'zebra': 'A zebra is in view. This is a horse-like African mammal with distinctive black and white stripes covering its entire body.',
      'giraffe': 'A giraffe is detected. This is the tallest mammal in the world, with an extremely long neck and legs, native to African savannas.',
      'backpack': 'A backpack is nearby. This is a bag with shoulder straps designed to be carried on someone\'s back, used for carrying personal items, books, or hiking gear.',
      'umbrella': 'An umbrella is present. This is a portable shelter consisting of a circular canopy on a folding frame, used for protection from rain or sun.',
      'handbag': 'A handbag is visible. This is a bag, typically carried by hand or over the shoulder, used for carrying personal items like wallets, keys, and phones.',
      'tie': 'A necktie is detected. This is a formal accessory worn around the neck, typically by men with dress shirts for business or formal occasions.',
      'suitcase': 'A suitcase is present. This is a rectangular traveling case with a handle, used for packing clothes and personal items when traveling.',
      'frisbee': 'A frisbee is nearby. This is a disc-shaped toy that is thrown and caught for recreation, often used in parks or beaches.',
      'skis': 'Skis are visible. These are long, narrow boards attached to boots for gliding over snow, used in winter sports and recreation.',
      'snowboard': 'A snowboard is detected. This is a wide board used for descending snow-covered slopes, similar to surfing but on snow.',
      'sports ball': 'A sports ball is in view. This is a round object used in various sports and games, could be a basketball, soccer ball, or similar.',
      'kite': 'A kite is visible. This is a lightweight object designed to fly in the wind, often diamond-shaped and controlled by a string from the ground.',
      'baseball bat': 'A baseball bat is nearby. This is a wooden or metal club used in baseball to hit the ball, typically about three feet long.',
      'baseball glove': 'A baseball glove is present. This is a leather mitt worn on the hand to catch and field baseballs during the game.',
      'skateboard': 'A skateboard is detected. This is a board with four wheels used for transportation and performing tricks, popular among youth.',
      'surfboard': 'A surfboard is visible. This is a long board used for riding ocean waves, a key piece of equipment in the sport of surfing.',
      'tennis racket': 'A tennis racket is nearby. This is a stringed paddle used to hit tennis balls, consisting of a handle and an oval head with strings.',
      'bottle': 'A bottle is present. This is a container, typically made of glass or plastic, used for storing and drinking liquids like water or beverages.',
      'wine glass': 'A wine glass is visible. This is a stemmed glass specifically designed for drinking wine, with a bowl, stem, and base.',
      'cup': 'A cup is detected. This is a small, open container used for drinking hot or cold beverages like coffee, tea, or water.',
      'fork': 'A fork is nearby. This is an eating utensil with two or more prongs, used for picking up and eating solid food.',
      'knife': 'A knife is present. This is a cutting tool with a sharp blade, used for cutting food during meals or food preparation.',
      'spoon': 'A spoon is visible. This is an eating utensil with a shallow bowl shape, used for eating liquids, soups, or soft foods.',
      'bowl': 'A bowl is detected. This is a round, deep container used for holding and eating food, especially soups, cereals, or salads.',
      'banana': 'A banana is present. This is a yellow, curved tropical fruit that is sweet, nutritious, and high in potassium and energy.',
      'apple': 'An apple is visible. This is a round fruit, typically red or green, that is sweet, crunchy, and rich in vitamins and fiber.',
      'sandwich': 'A sandwich is nearby. This is a food item consisting of ingredients like meat, cheese, or vegetables placed between slices of bread.',
      'orange': 'An orange is detected. This is a round, orange-colored citrus fruit that is juicy, sweet, and high in vitamin C.',
      'broccoli': 'Broccoli is in view. This is a green vegetable with a tree-like appearance, known for being very nutritious and high in vitamins.',
      'carrot': 'A carrot is present. This is an orange root vegetable that is long and tapered, sweet in taste and rich in beta-carotene.',
      'hot dog': 'A hot dog is visible. This is a cooked sausage typically served in a split bun with various condiments like mustard or ketchup.',
      'pizza': 'Pizza is detected. This is an Italian dish consisting of a flat bread base topped with tomato sauce, cheese, and various toppings.',
      'donut': 'A donut is nearby. This is a sweet, fried pastry with a ring shape, often glazed or covered with sugar and various toppings.',
      'cake': 'A cake is present. This is a sweet baked dessert, often layered and decorated with frosting, typically served at celebrations.',
      'chair': 'A chair is visible. This is a piece of furniture designed for one person to sit on, with a back support and usually four legs.',
      'couch': 'A couch is detected. This is upholstered furniture designed for multiple people to sit on, also called a sofa, commonly found in living rooms.',
      'potted plant': 'A potted plant is nearby. This is vegetation growing in a container or pot, used for decoration or air purification indoors.',
      'bed': 'A bed is present. This is furniture designed for sleeping and resting, typically consisting of a mattress on a frame with pillows.',
      'dining table': 'A dining table is visible. This is furniture used for eating meals, where people sit around to share food and conversation.',
      'toilet': 'A toilet is detected. This is a plumbing fixture used for waste disposal, typically found in bathrooms and restrooms.',
      'tv': 'A television is nearby. This is an electronic device with a screen that displays video content, news, movies, and entertainment programs.',
      'laptop': 'A laptop is present. This is a portable computer that can be folded shut, allowing you to work, browse internet, or watch videos anywhere.',
      'mouse': 'A computer mouse is visible. This is a pointing device used to control the cursor on a computer screen by moving it on a surface.',
      'remote': 'A remote control is detected. This is a handheld device used to operate electronic equipment like televisions or stereos from a distance.',
      'keyboard': 'A keyboard is nearby. This is an input device with keys for typing letters, numbers, and commands into a computer or device.',
      'cell phone': 'A cell phone is present. This is a mobile communication device that allows you to make calls, send messages, and access the internet.',
      'microwave': 'A microwave is visible. This is a kitchen appliance that heats food quickly using electromagnetic waves called microwaves.',
      'oven': 'An oven is detected. This is a kitchen appliance used for baking, roasting, or heating food using dry heat in an enclosed space.',
      'toaster': 'A toaster is nearby. This is a small kitchen appliance designed to brown bread slices by exposing them to radiant heat.',
      'sink': 'A sink is present. This is a basin with faucets used for washing dishes, hands, or food preparation in kitchens and bathrooms.',
      'refrigerator': 'A refrigerator is visible. This is a large kitchen appliance that keeps food and beverages cold and fresh for longer storage.',
      'book': 'A book is detected. This is a bound collection of printed pages containing written content like stories, information, or knowledge.',
      'clock': 'A clock is nearby. This is a device that displays the current time, helping people keep track of hours and minutes throughout the day.',
      'vase': 'A vase is present. This is a decorative container, often made of glass or ceramic, typically used for holding flowers or as decoration.',
      'scissors': 'Scissors are visible. This is a cutting tool with two sharp blades that pivot against each other, used for cutting paper, fabric, or other materials.',
      'teddy bear': 'A teddy bear is detected. This is a soft stuffed toy designed to resemble a bear, commonly given to children for comfort and play.',
      'hair drier': 'A hair dryer is nearby. This is an electrical device that blows hot air to dry wet hair quickly after washing or styling.',
      'toothbrush': 'A toothbrush is present. This is a small brush with bristles used for cleaning teeth and maintaining oral hygiene, typically used with toothpaste.'
    };
    
    return descriptions[label] || `A ${label} is detected in your view. This object is part of your current environment and may require your attention.`;
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