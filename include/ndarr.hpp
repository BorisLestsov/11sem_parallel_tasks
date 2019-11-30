
class NdArr{
public:

    NdArr(const std::vector<int>& shape_, float value=0):
        Nd(shape_.size()),
        shape(shape_),
        strides(shape.size(), 1)
    {
        arr_size = 1;
        for (int i = 0; i < shape_.size(); ++i){
            arr_size *= shape_[i];
        }
        strides[shape_.size()-1] = 1;
        for (int i = shape_.size()-2; i >= 0; --i){
            strides[i] = strides[i+1]*shape_[i+1];
        }
        // for (int i = 0; i < shape_.size(); ++i){
        //     std::cout << strides[i] << std::endl;
        // }

        // Default float constructor is zero-initialize
        arr = new float[arr_size];

        for (int i = 0; i < arr_size; ++i){
            arr[i] = value;
        }
    }

    ~NdArr(){
        delete []arr;
    }

    float& operator()(int i){
        return arr[i];
    }

    float& operator()(int i, int j){
        return arr[i*strides[0] + j];
    }

    float& operator()(int i, int j, int k){
        return arr[i*strides[0] + j*strides[1] + k];
    }

    float& operator()(int i, int j, int k, int l){
        return arr[i*strides[0] + j*strides[1] + k*strides[2] + l];
    }

    int size(){
        return arr_size;
    }


    float* arr;
    int Nd;
    std::vector<int> shape;
    std::vector<int> strides;
    int arr_size;

};
