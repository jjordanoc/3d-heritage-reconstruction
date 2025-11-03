# WebSocket Server Deployment Guide

## Quick Start

### 1. Deploy the WebSocket Server

```bash
cd backend
modal deploy websocket_server.py
```

This will deploy the server and give you a URL like:
```
https://your-username--ut3c-heritage-websocket.modal.run
```

### 2. Test the Connection

#### Option A: Use the HTML Test Client

1. Open `websocket_client_example.html` in your browser
2. Update the WebSocket URL to your deployed Modal URL
3. Change `ws://localhost:8000` to `wss://your-username--ut3c-heritage-websocket.modal.run`
4. Click "Connect"
5. Upload an image to test

#### Option B: Use curl for HTTP endpoints

```bash
# Health check
curl https://your-username--ut3c-heritage-websocket.modal.run/health

# Project status
curl https://your-username--ut3c-heritage-websocket.modal.run/projects/auditorio/status
```

### 3. Replace black_box Stub

Once your colleague finishes the `black_box` function, replace the stub:

```python
# In websocket_server.py, the import will automatically use the real implementation:
from black_box import black_box
```

Make sure the `black_box` function has this signature:

```python
def black_box(input_img_path: str, old_point_cloud: str, new_img_id: str) -> str:
    """
    Returns path to folder containing PLY shard files
    """
    # Your colleague's implementation here
    return "/path/to/shard/folder"
```

## Architecture

### Data Flow

```
Client 1                Server                  Client 2
   |                       |                       |
   |----[upload_image]---->|                       |
   |                       |                       |
   |<--[acknowledged]------|                       |
   |                       |                       |
   |                   [save image]                |
   |                   [call black_box]            |
   |                   [get shard folder]          |
   |                       |                       |
   |<---[shard_update]-----|----[shard_update]---->|
   |                       |                       |
```

### State Management

The server maintains state per project:
- **update_count**: Increments with each image (resets at 10)
- **total_images**: Total images processed
- **last_full_refetch**: Timestamp of last full reconstruction

### Full Refetch Trigger

Every 10th image triggers a full refetch:
- Images 1-9: Incremental updates (single shard)
- Image 10: Full refetch (all shards, counter resets)
- Images 11-19: Incremental updates
- Image 20: Full refetch (all shards, counter resets)
- And so on...

## Integration with Existing Server

Both servers can run simultaneously:

| Feature | Old Server (`modal_server.py`) | New Server (`websocket_server.py`) |
|---------|-------------------------------|-----------------------------------|
| Purpose | Traditional HTTP endpoints | Real-time WebSocket streaming |
| Upload | POST `/pointcloud/{id}` | WebSocket message |
| Response | Full point cloud in response | Streamed shards |
| Multi-user | No broadcasting | Broadcasts to all connected clients |
| Volume | `ut3c-heritage` | `ut3c-heritage` (same) |
| Modal App | `ut3c-heritage-backend-brute` | `ut3c-heritage-websocket` |

They share the same volume, so files written by one can be read by the other.

## Frontend Integration

### JavaScript/TypeScript Example

```typescript
class HeritageReconstructionClient {
  private ws: WebSocket;
  private projectId: string;
  
  constructor(wsUrl: string, projectId: string) {
    this.projectId = projectId;
    this.ws = new WebSocket(`${wsUrl}/ws/${projectId}`);
    this.setupHandlers();
  }
  
  private setupHandlers() {
    this.ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      this.handleMessage(message);
    };
  }
  
  private handleMessage(message: any) {
    switch (message.type) {
      case 'shard_update':
        this.loadPLYShard(message.shard_data, message.is_full_refetch);
        break;
      case 'refetch_starting':
        this.clearPointClouds();
        break;
      // ... handle other message types
    }
  }
  
  private loadPLYShard(base64Data: string, isFullRefetch: boolean) {
    // Decode base64
    const binaryString = atob(base64Data);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    
    // Load into your 3D viewer (Three.js, Babylon.js, etc.)
    // ...
  }
  
  uploadImage(file: File) {
    const reader = new FileReader();
    reader.onload = () => {
      const base64 = reader.result.split(',')[1];
      this.ws.send(JSON.stringify({
        type: 'upload_image',
        image_data: base64
      }));
    };
    reader.readAsDataURL(file);
  }
}

// Usage
const client = new HeritageReconstructionClient(
  'wss://your-modal-url.modal.run',
  'auditorio'
);

// Upload when user selects a file
fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  client.uploadImage(file);
});
```

### Vue.js Integration

```vue
<script setup>
import { ref, onMounted, onUnmounted } from 'vue'

const ws = ref(null)
const pointClouds = ref([])

onMounted(() => {
  ws.value = new WebSocket('wss://your-modal-url.modal.run/ws/auditorio')
  
  ws.value.onmessage = (event) => {
    const message = JSON.parse(event.data)
    
    if (message.type === 'shard_update') {
      const plyData = atob(message.shard_data)
      pointClouds.value.push(plyData)
      // Load into your PLYViewer component
    }
  }
})

onUnmounted(() => {
  ws.value?.close()
})

const uploadImage = (file) => {
  const reader = new FileReader()
  reader.onload = () => {
    const base64 = reader.result.split(',')[1]
    ws.value.send(JSON.stringify({
      type: 'upload_image',
      image_data: base64
    }))
  }
  reader.readAsDataURL(file)
}
</script>
```

## Testing Multi-User

To test multi-user functionality:

1. Open `websocket_client_example.html` in two different browser windows/tabs
2. Connect both to the same project ID (e.g., "auditorio")
3. Upload an image from one client
4. Watch as both clients receive the update in real-time

## Monitoring

### Check Active Connections

```bash
curl https://your-modal-url.modal.run/health
```

Response:
```json
{
  "status": "healthy",
  "service": "heritage-reconstruction-websocket",
  "active_projects": 2,
  "total_connections": 5
}
```

### Check Project Status

```bash
curl https://your-modal-url.modal.run/projects/auditorio/status
```

Response:
```json
{
  "project_id": "auditorio",
  "active_clients": 3,
  "update_count": 7,
  "total_images": 27,
  "last_full_refetch": "2025-11-02T14:30:00"
}
```

## Troubleshooting

### Connection Fails

**Problem**: WebSocket connection fails or immediately closes

**Solutions**:
1. Make sure you're using `wss://` (secure WebSocket) for HTTPS deployments
2. Check CORS settings in the server
3. Verify the Modal app is deployed and running: `modal app list`

### No Shards Received

**Problem**: Connected but not receiving point cloud shards

**Solutions**:
1. Check that `black_box` function is implemented
2. Verify Modal volume is mounted correctly
3. Check server logs: `modal app logs ut3c-heritage-websocket`

### Memory Issues

**Problem**: Server runs out of memory with many clients

**Solutions**:
1. Reduce shard size by downsampling point clouds
2. Implement voxel downsampling in `black_box`
3. Consider Redis for state management (for scaling)

## Performance Tips

1. **Compress PLY files**: Use gzip before base64 encoding
2. **Downsample**: Reduce point cloud density for mobile clients
3. **Batch shards**: Combine small shards to reduce message count
4. **Client-side caching**: Cache reference set on client to survive reconnections

## Security Considerations

Before production:

1. **Authentication**: Add project-level authentication
2. **Rate limiting**: Prevent upload spam
3. **File validation**: Verify uploaded images
4. **CORS**: Restrict to your domain
5. **Size limits**: Enforce max file size

## Next Steps

1. ✅ Deploy the WebSocket server
2. ✅ Test with HTML client
3. ⏳ Wait for `black_box` implementation
4. ⏳ Integrate with your Vue.js frontend
5. ⏳ Test multi-user collaboration
6. ⏳ Deploy to production with security measures

