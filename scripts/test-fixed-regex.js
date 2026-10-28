const regex = /ncclCommInitAll\s*\(\s*&?\s*([\w.\[\]]+)\s*,\s*([^,)]+)(?:,[^)]*)?\)/;

const testInput = `
    int nGpus = 2;
    int count = 4;
    ncclComm_t comms[2];
    ncclCommInitAll(comms, 2);

    float *d_send[2];
    float *d_recv[2];
`;

console.log('Fixed regex match:');
console.log(testInput.match(regex));
