#include <stdio.h>
#include <time.h>
#include <stdint.h>
#define rnd( x ) (x * rand() / RAND_MAX)
#define INF     2e10f
//asdf
#define _bitsperpixel 32
#define _planes 1
#define _compression 0

#define _xpixelpermeter 0x13B //0x130B //2835 , 72 DPI
#define _ypixelpermeter 0x13B//0x130B //2835 , 72 DPI
#define pixel 0xFF
#pragma pack(push,1)
typedef struct{
    uint8_t signature[2];
    uint32_t filesize;
    uint32_t reserved;
    uint32_t fileoffset_to_pixelarray;
} fileheader;
typedef struct{
    uint32_t dibheadersize;
    uint32_t width;
    uint32_t height;
    uint16_t planes;
    uint16_t bitsperpixel;
    uint32_t compression;
    uint32_t imagesize;
    uint32_t ypixelpermeter;
    uint32_t xpixelpermeter;
    uint32_t numcolorspallette;
    uint32_t mostimpcolor;
} bitmapinfoheader;
typedef struct {
    fileheader fileheader;
    bitmapinfoheader bitmapinfoheader;
} bitmap;
#pragma pack(pop)
void genBpm (int height, int width, int *pixelbuffer_f) {
    uint32_t pixelbytesize = height*width*_bitsperpixel/8;
    uint32_t  _filesize =pixelbytesize+sizeof(bitmap);
    FILE *fp = fopen("raycuda.bmp","wb");
    bitmap *pbitmap  = (bitmap*)calloc(1,sizeof(bitmap));

    int buffer_size = height*width*4;
    uint8_t *pixelbuffer = (uint8_t*)malloc(buffer_size);

    for(int i = 0; i<buffer_size;i++)
    {
     pixelbuffer[i]= (uint8_t) pixelbuffer_f[i];
    }


    //strcpy(pbitmap->fileheader.signature,"BM");
    pbitmap->fileheader.signature[0] = 'B';
    pbitmap->fileheader.signature[1] = 'M';
    pbitmap->fileheader.filesize = _filesize;
    pbitmap->fileheader.fileoffset_to_pixelarray = sizeof(bitmap);
    pbitmap->bitmapinfoheader.dibheadersize =sizeof(bitmapinfoheader);
    pbitmap->bitmapinfoheader.width = width;
    pbitmap->bitmapinfoheader.height = height;
    pbitmap->bitmapinfoheader.planes = _planes;
    pbitmap->bitmapinfoheader.bitsperpixel = _bitsperpixel;
    pbitmap->bitmapinfoheader.compression = _compression;
    pbitmap->bitmapinfoheader.imagesize = pixelbytesize;
    pbitmap->bitmapinfoheader.ypixelpermeter = _ypixelpermeter ;
    pbitmap->bitmapinfoheader.xpixelpermeter = _xpixelpermeter ;
    pbitmap->bitmapinfoheader.numcolorspallette = 0;
    fwrite (pbitmap, 1, sizeof(bitmap),fp);
    //memset(pixelbuffer,pixel,pixelbytesize);
    fwrite(pixelbuffer,1,pixelbytesize,fp);
    fclose(fp);
    free(pbitmap);
    free(pixelbuffer);
}


struct Sphere {
    float   r,b,g;
    float   radius;
    float   x,y,z;
};

void loadSpheres(Sphere *vet, int size, float dim, float radius, float sum){
   
	for (int i=0;i<size;i++){
			Sphere sphere;
            sphere.r = rnd(1.0);
            sphere.b = rnd(1.0);
            sphere.g = rnd(1.0);
            sphere.radius = rnd( 100.0f ) + 20;
            sphere.x = rnd( 1000.0f ) - 500;
            sphere.y = rnd( 1000.0f ) - 500;
            sphere.z = rnd( 1000.0f ) - 500;

            vet[i] = sphere;
            
           
        }
}

#define SPHERES 20

__global__ void kernel(int dim, Sphere * s,  int *ptr ) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    float   ox = (x - dim/2);
    float   oy = (y - dim/2);

    float   r=0, g=0, b=0;
    float   maxz = -99999;
    for(int i=0; i<SPHERES; i++) {
        float   n;
        float   t = -99999;
        float dx = ox - s[i].x;
        float dy = oy - s[i].y;
        float dz;
        if (dx*dx + dy*dy < s[i].radius * s[i].radius) {
            dz = sqrtf( s[i].radius * s[i].radius - dx*dx - dy*dy );
            n = dz / sqrtf( s[i].radius * s[i].radius );
            t = dz + s[i].z;

        } else {
            t = -99999;
        }
        if (t > maxz) {
              float fscale = n;
              r = s[i].r * fscale;
              g = s[i].g * fscale;
              b = s[i].b * fscale;
              maxz = t;
        }

    }

    ptr[offset*4 + 0] = (r * 255);
    ptr[offset*4 + 1] = (g * 255);
    ptr[offset*4 + 2] = (b * 255);
    ptr[offset*4 + 3] = 255;
}


int main(int argc, char *argv[]){
    int dim = atoi(argv[1]);
    //int sph = atoi(argv[2]);
    
    int  *final_image;
    int   *dev_image;
    Sphere * s;

    final_image = (int*) malloc(dim * dim * sizeof(int)*4);
    Sphere *temp_s = (Sphere*)malloc( sizeof(Sphere) * SPHERES );
    printf("before load sphres");
    loadSpheres(temp_s, SPHERES, dim, 160, 20);

   printf("before\n");

    float time;
    cudaEvent_t start, stop;   
    cudaEventCreate(&start) ;
    cudaEventCreate(&stop) ;
    cudaEventRecord(start, 0) ;

    cudaMalloc( (void**)&dev_image, dim * dim * sizeof(int)*4);
    cudaMalloc( (void**)&s, sizeof(Sphere) * SPHERES );
    
    cudaMemcpy( s, temp_s, sizeof(Sphere) * SPHERES, cudaMemcpyHostToDevice );
   printf("after alloc\n");
    dim3    grids(dim/16,dim/16);
    dim3    threads(16,16);

    kernel<<<grids,threads>>>(dim, s, dev_image);
   printf("after kernel\n");

    cudaMemcpy( final_image, dev_image, dim * dim * sizeof(int) * 4,cudaMemcpyDeviceToHost );
        printf("after copy back\n");
   
    cudaFree( dev_image);
    cudaFree( s );
    
    cudaEventRecord(stop, 0) ;
    cudaEventSynchronize(stop) ;
    cudaEventElapsedTime(&time, start, stop) ;

     printf("CUDA\t%d\t%3.1f\n", dim,time);
     genBpm(dim,dim,final_image);
    
    free(temp_s);
    free(final_image);


}