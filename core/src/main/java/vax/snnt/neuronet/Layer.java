package vax.snnt.neuronet;

import java.util.LinkedList;
import java.util.List;

/**

 @author toor
 @param <T>
 */
public class Layer<T extends Neuron> implements Processable {
    private final List<T> neurons = new LinkedList<>();

    /*
     public Layer ( List<Neuron> neurons ) {
     this.neurons = neurons;
     }
     */

 /*
     public Layer ( Layer nextLayer ) {

     }
     */
    public Layer () {
    }

    public void addNeuron ( T neuron ) {
        neurons.add( neuron );
    }

    @Override
    public void process () {
        for( T neuron : neurons ) {
            neuron.process();
        }
    }

    public List<T> getNeurons () {
        return neurons;
    }

}
