# WebSocket Heritage Reconstruction Server

Real-time, multi-user 3D heritage reconstruction with WebSocket streaming.

## Architecture Overview

This server provides WebSocket-based streaming for collaborative 3D reconstruction:

- **Multi-user support**: Multiple clients can view the same heritage site simultaneously
- **Incremental updates**: New reconstructions are streamed as small aligned point clouds
- **Periodic full refetch**: Every 10th image triggers a complete reconstruction
- **Progressive loading**: Large point clouds are streamed as shards for mobile support
- **New client sync**: Connecting clients receive the current reference set

## Files

- `websocket_server.py` - Main WebSocket server (Modal app)
- `black_box_stub.py` - Placeholder for the reconstruction function
- `WEBSOCKET_README.md` - This file
- `websocket_client_example.html` - Simple HTML/JS test client

## How It Works

### 1. Client Connection

Clients connect to `/ws/{project_id}` where `project_id` identifies a heritage site (e.g., "auditorio", "pucllana").

```javascript
const ws = new WebSocket('wss://your-modal-url/ws/auditorio');
```

Upon connection, the server sends the current reference set (all shards from the last full reconstruction).

### 2. Image Upload

Clients send images as base64-encoded data:

```json
{
  "type": "upload_image",
  "image_data": "<base64_encoded_image>"
}
```

### 3. Reconstruction Flow

**Incremental updates (images 1-9, 11-19, etc.):**
1. Server saves image to volume
2. Calls `black_box(img_path, reference_cloud_path, img_id)`
3. black_box returns folder with aligned point cloud shard(s)
4. Server streams shard(s) to ALL connected clients
5. Increment update counter

**Full refetch (every 10th image: 10, 20, 30, etc.):**
1. Server saves image
2. Calls `black_box` which performs full reconstruction
3. black_box returns folder with ALL shards (new reference set)
4. Server sends `refetch_starting` notification
5. Server streams all shards to ALL connected clients
6. Reset update counter to 0

### 4. Message Protocol

#### Client → Server

**Upload Image**
```json
{
  "type": "upload_image",
  "image_data": "<base64_encoded_png_or_jpg>"
}
```

**Request Resync**
```json
{
  "type": "request_resync"
}
```

**Ping (keepalive)**
```json
{
  "type": "ping"
}
```

#### Server → Client

**Initial State - Starting**
```json
{
  "type": "initial_state_starting",
  "total_shards": 15,
  "total_images": 50
}
```

**Initial State - Shard**
```json
{
  "type": "initial_state_shard",
  "shard_index": 0,
  "total_shards": 15,
  "shard_data": "<base64_encoded_ply>"
}
```

**Initial State - Complete**
```json
{
  "type": "initial_state_complete",
  "total_images": 50
}
```

**Upload Acknowledged**
```json
{
  "type": "upload_acknowledged",
  "image_id": "uuid-here",
  "message": "Image received, processing..."
}
```

**Processing Started**
```json
{
  "type": "processing_started",
  "image_id": "uuid-here"
}
```

**Refetch Starting** (only for full refetch)
```json
{
  "type": "refetch_starting",
  "reason": "10 updates reached - full reconstruction",
  "total_shards": 25,
  "sequence": 10
}
```

**Shard Update**
```json
{
  "type": "shard_update",
  "sequence": 5,
  "shard_index": 0,
  "total_shards": 3,
  "is_full_refetch": false,
  "shard_data": "<base64_encoded_ply>",
  "timestamp": "2025-11-02T12:34:56.789"
}
```

**Processing Complete**
```json
{
  "type": "processing_complete",
  "image_id": "uuid-here",
  "sequence": 5,
  "total_images": 55
}
```

**Error**
```json
{
  "type": "error",
  "message": "Error description"
}
```

**Pong**
```json
{
  "type": "pong",
  "timestamp": "2025-11-02T12:34:56.789"
}
```

## State Management

The server maintains per-project state:

```python
{
  "update_count": 5,           # Number of updates since last full refetch
  "last_full_refetch": "2025-11-02T12:00:00",
  "total_images": 55           # Total images in reconstruction
}
```

- **update_count** increments with each image
- When **update_count** reaches 10, trigger full refetch and reset to 0
- **total_images** tracks cumulative images processed

## Integration with black_box

The `black_box` function (implemented by another team member) has this signature:

```python
def black_box(input_img_path: str, old_point_cloud: str, new_img_id: str) -> str:
    """
    Performs reconstruction and alignment.
    
    Args:
        input_img_path: Path to the new preprocessed image
        old_point_cloud: Path to the reference point cloud (latest.ply)
        new_img_id: Unique identifier for this image
    
    Returns:
        Path to folder containing PLY shard files:
        - Incremental: Single shard with aligned new point cloud
        - Full refetch: Multiple shards (entire reconstruction)
    """
```

## Deployment

### Deploy to Modal

```bash
cd backend
modal deploy websocket_server.py
```

This will output a URL like: `https://your-username--ut3c-heritage-websocket.modal.run`

### Test the Server

```bash
# Health check
curl https://your-modal-url/health

# Project status
curl https://your-modal-url/projects/auditorio/status
```

### Connect with WebSocket

Use the provided `websocket_client_example.html` or integrate into your frontend:

```javascript
const ws = new WebSocket('wss://your-modal-url/ws/auditorio');

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  
  switch (message.type) {
    case 'shard_update':
      // Decode base64 PLY and add to scene
      const plyBytes = atob(message.shard_data);
      // ... load into Three.js or your 3D viewer
      break;
      
    case 'refetch_starting':
      // Clear existing point clouds, prepare for full reload
      break;
      
    // ... handle other message types
  }
};

// Upload an image
function uploadImage(imageFile) {
  const reader = new FileReader();
  reader.onload = () => {
    const base64 = reader.result.split(',')[1]; // Remove data:image/png;base64, prefix
    ws.send(JSON.stringify({
      type: 'upload_image',
      image_data: base64
    }));
  };
  reader.readAsDataURL(imageFile);
}
```

## HTTP Endpoints

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "heritage-reconstruction-websocket",
  "active_projects": 2,
  "total_connections": 5
}
```

### GET /projects/{project_id}/status

Get status of a specific project.

**Response:**
```json
{
  "project_id": "auditorio",
  "active_clients": 3,
  "update_count": 5,
  "total_images": 55,
  "last_full_refetch": "2025-11-02T12:00:00"
}
```

## Key Features

### ✅ Multi-User Broadcasting
- All clients viewing the same heritage site receive updates simultaneously
- Perfect for collaborative reconstruction sessions

### ✅ Incremental Updates
- Only new aligned point clouds are streamed (not the entire reconstruction)
- Reduces bandwidth and improves mobile performance

### ✅ Progressive Loading
- Large reconstructions are split into shards
- Clients can display partial results while loading continues

### ✅ Automatic Refetch
- Every 10th image triggers a full reconstruction
- Prevents drift from accumulated alignment errors

### ✅ New Client Sync
- Connecting clients automatically receive the current reference set
- No manual synchronization required

### ✅ Error Recovery
- Clients can request resync if they miss updates
- Graceful handling of disconnections and reconnections

## Coordination with Existing Server

This WebSocket server is **completely separate** from the existing `modal_server.py`:

- **Different Modal app name**: `ut3c-heritage-websocket` vs `ut3c-heritage-backend-brute`
- **Same volume**: Both use `ut3c-heritage` volume for data sharing
- **Coexistence**: Both servers can run simultaneously
- **No conflicts**: WebSocket server doesn't modify existing endpoints

Clients can choose:
- **Legacy mode**: Use POST `/pointcloud/{id}` endpoint (existing server)
- **Real-time mode**: Use WebSocket `/ws/{project_id}` (this server)

## Testing Strategy

1. **Single client test**: Upload image, verify shard streaming
2. **Multi-client test**: Connect 2+ clients, verify all receive updates
3. **Full refetch test**: Upload 10 images, verify refetch triggers and counter resets
4. **Reconnection test**: Disconnect and reconnect, verify initial state sync
5. **Mobile test**: Test on actual mobile device with limited bandwidth

## Performance Considerations

- **Shard size**: Keep individual PLY shards under 1MB for mobile
- **Streaming delay**: 100ms delay between shards prevents overwhelming clients
- **Base64 overhead**: ~33% size increase, consider binary WebSocket frames for production
- **Memory**: Modal containers handle state in-memory; consider Redis for scaling

## Future Enhancements

- [ ] Binary WebSocket frames (more efficient than base64 JSON)
- [ ] Compression (gzip PLY data before base64 encoding)
- [ ] Redis pub/sub for multi-container scaling
- [ ] Authentication/authorization per project
- [ ] Client-side downsampling hints based on network quality
- [ ] Sequence number acknowledgments for guaranteed delivery

