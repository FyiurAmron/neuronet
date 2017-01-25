package vax.snnt.neuronet;

import static java.lang.Double.isNaN;
import static java.lang.Math.abs;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

/**

 @author toor
 */
public class Network implements Processable {
    private final LinkedList<Layer<? extends Neuron>> layers = new LinkedList<>();
    private final ArrayList<Double> outputs = new ArrayList<>();
    private double distL1, distL2;

    public Network () {
    }

    public void addLayer ( Layer<? extends Neuron> layer ) {
        layers.add( layer );
    }

    /*
     public Layer<? extends Neuron> getInputLayer() {
     return layers.getFirst();
     }
     */
    public void addInputs ( double[] inputs ) {
        List<? extends Neuron> inputNeurons = layers.getFirst().getNeurons();
        if ( inputNeurons.size() != inputs.length ) {
            throw new IllegalArgumentException();
        }
        int i = 0;
        for( Neuron neuron : inputNeurons ) {
            neuron.add( inputs[i] );
            i++;
        }
    }

    public void collectOutput ( double output ) {
        outputs.add( output );
    }

    public double[] getOutputs () {
        int count = outputs.size();
        double[] ret = new double[count];
        for( int i = 0; i < count; i++ ) {
            ret[i] = outputs.get( i );
        }
        return ret;
    }

    @Override
    public void process () {
        outputs.clear();
        for( Layer<? extends Neuron> layer : layers ) {
            layer.process();
        }
    }

    public void calcOutputDist ( double[] expectedOutputs ) {
        int count = outputs.size();
        if ( expectedOutputs.length != count ) {
            throw new IllegalArgumentException();
        }

        for( int i = 0; i < count; i++ ) {
            double diff = outputs.get( i ) - expectedOutputs[i];
            distL1 += abs( diff );
            distL2 += diff * diff;
        }
        if ( isNaN( distL1 ) || isNaN( distL2 ) ) {
            throw new RuntimeException();
        }
    }

    public void resetOutputDist () {
        distL1 = 0;
        distL2 = 0;
    }

    public double getDistL1 () {
        return distL1;
    }

    public double getDistL2 () {
        return distL2;
    }

}
