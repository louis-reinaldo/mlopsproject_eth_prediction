from batch_score import score


def test_score_flow():

    """Check if scoring flow is completed"""
    result = score(return_state = True)
    assert dict(result)['name'] == 'Completed'

if __name__ == "__main__": 
    test_score_flow()