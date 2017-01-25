package vax.snnt.neuronet;

/**

 @author toor
 */
public class TestNetwork extends Network {
    int layerCount = 4;
    int neuronsPerLayer = 10;
    double randomAmpMin = -Math.E;
    double randomAmpMax = +Math.E;
    int inputCount = 1;
    int outputCount = 1;

    public TestNetwork () {
        /*
         List<Neuron> inputLayer = new ArrayList<>( inputCount );
         List<Neuron> outputLayer = new ArrayList<>( outputCount );

         for( int i = 0; i < outputCount; i++ ) {
         //Neuron output = new Neuron( null, inputTransferFunction, outputTransferFunction );
         }
         */

    }
}
